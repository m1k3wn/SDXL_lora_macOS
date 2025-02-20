import logging
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import safetensors
from safetensors.torch import save_file
from tqdm.auto import tqdm
from config import ProjectConfig
from typing import Dict, List, Optional, Tuple, Union

class LoRADataset(Dataset):
    def __init__(self, config: ProjectConfig, dataset_type: str):
        self.config = config
        self.dataset_type = dataset_type
        self.image_dir = config.paths['dataset'][dataset_type]
        self.metadata_path = config.paths['metadata'][dataset_type]
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize((512 if dataset_type == "test" else 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.metadata['files'])
    
    def __getitem__(self, idx):
        try:
            item = self.metadata['files'][idx]
            image_path = self.image_dir / item['filename']
            
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            style_config = self.config.get_training_config()['style']
            base_prompt = style_config['base_prompt'].format(trigger_word=style_config['trigger_word'])
            prompt = f"{item['caption']}, {base_prompt}"
            
            return {
                'image': image,
                'prompt': prompt,
                'negative_prompt': style_config['negative_prompt']
            }
        except Exception as e:
            self.logger.error(f"Error loading item {idx}: {str(e)}")
            raise

class LoRATrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("mps")
        self.load_model()
        self.setup_accelerator()
        self.setup_tokenizer()
        self.setup_datasets()

        # Fix: Swap the encoders to match SDXL expectations
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            self.config.paths["base_model"], subfolder="text_encoder_2"
        ).to(self.device)

        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.config.paths["base_model"], subfolder="text_encoder"
        ).to(self.device)

                
    def setup_tokenizer(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.paths['base_model'],
            subfolder="tokenizer"
        )
        
    def setup_accelerator(self):
        self.device = torch.device("mps")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get_training_config()['training']['gradient_accumulation_steps'],
            mixed_precision="no", 
            device_placement=True
        )
        
        self.model.to(self.device)

    # Loads model and text encoders         
    def load_model(self):
        self.model = StableDiffusionXLPipeline.from_pretrained(
            self.config.paths["base_model"],
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config
        )
        
        self.text_encoder_1 = self.model.text_encoder
        self.text_encoder_2 = self.model.text_encoder_2

        self.vae = self.model.vae.to(self.device)
        self.noise_scheduler = self.model.scheduler
            
    def setup_datasets(self):
        # If in test mode, use `test_dataset` for both training and validation
        if self.config.mode == "test":
            self.logger.info("Test mode active: Using 512px test dataset for both training and validation.")
            self.train_dataset = LoRADataset(self.config, "test")  # Use test dataset for training
            self.test_dataset = LoRADataset(self.config, "test")
        else:
            self.logger.info("Production mode active: Using 1024px train dataset for both training and validation.")
            self.train_dataset = LoRADataset(self.config, "train")
            self.test_dataset = LoRADataset(self.config, "test")

        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get_training_config()['training']['batch_size'],
            shuffle=True
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False
        )
        
    # handles SDXL's requirement for aspect/crop conditioning
    def _get_add_time_ids(self, batch, dtype):
        original_size = (1024, 1024)
        if self.config.mode == "test":
            original_size = (512, 512)  # Handle test mode size
        target_size = original_size
        crops_coords_top_left = (0, 0)
        
        add_time_ids = torch.tensor([
            target_size[0] / original_size[0],
            target_size[1] / original_size[1],
            crops_coords_top_left[0] / original_size[0],
            crops_coords_top_left[1] / original_size[1],
        ], device=self.device, dtype=dtype)
        
        add_time_ids = add_time_ids.unsqueeze(0).repeat(batch["image"].shape[0], 1)
        return add_time_ids   
    
    # configures LoRA params
    def setup_lora(self):
        """Configure LoRA layers for SDXL"""
        config = self.config.get_training_config()['lora']
        
        self.unet = self.model.unet
        self.unet.enable_lora = True
        self.unet.lora_rank = config['rank']
        self.unet.lora_alpha = config['alpha'] 
        self.unet.lora_dropout = config['dropout'] 
        self.unet.train()
                
    def train(self):
        training_config = self.config.get_training_config()['training']
        
        # Setup LoRA
        self.setup_lora()
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        # Move VAE to eval as it isn't trained
        self.vae.eval()  

        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=training_config['learning_rate']
        )
        
        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            mixed_precision="fp16",
            device_placement=True
        )
        
        # Prepare models, optimizer and dataloader
        self.unet, self.text_encoder_1, self.text_encoder_2, optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet, self.text_encoder_1, self.text_encoder_2, optimizer, self.train_dataloader
        )
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        global_step = 0
        for epoch in range(training_config['num_epochs']):
            self.model.unet.train()
            progress_bar = tqdm(total=len(self.train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.unet):
                    # Get text embeddings
                    text_inputs = self.tokenizer(
                        batch["prompt"], 
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    # Generate text embeddings with no gradients 
                    with torch.no_grad():
                        text_embeds_1 = self.text_encoder_1(text_inputs.input_ids)[0]
                        pooled_embeds_1 = self.text_encoder_1(text_inputs.input_ids)[1]
                        text_embeds_2 = self.text_encoder_2(text_inputs.input_ids)[0]
                        pooled_embeds_2 = self.text_encoder_2(text_inputs.input_ids)[1]

                    # Concatenate embeddings properly for SDXL
                    text_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
                    pooled_embeds = torch.cat([pooled_embeds_1, pooled_embeds_2], dim=-1)

                    # Add time embeddings
                    add_time_ids = self._get_add_time_ids(batch, text_embeds.dtype)
                    added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

                    # Convert images to latents
                    latents = self.vae.encode(batch["image"].to(self.device)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],), device=self.device)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get model prediction and compute loss
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample

                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                global_step += 1
                
                if global_step % training_config['save_steps'] == 0:
                    self.save_model(f"checkpoint-{global_step}")
                    
                if global_step % training_config['validation_steps'] == 0:
                    self.validate()

            # Memory cleanup at end of each epoch        
            if torch.cuda.is_available() or hasattr(torch.backends, 'mps'):
                torch.cuda.empty_cache()

            progress_bar.close()
            
        # Final save
        self.save_model("final")
        
    def save_model(self, tag: str):
        save_path = self.config.paths['output'] / f"lora_{tag}.safetensors"
        
        # Get unwrapped unet
        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        
        # Extract LoRA state dict
        lora_state_dict = {}
        for key, value in unwrapped_unet.state_dict().items():
            if "lora" in key:
                lora_state_dict[key] = value.detach().cpu()
        
        # Save with safetensors
        safetensors.torch.save_file(lora_state_dict, save_path)
            
    def validate(self):
        self.unet.eval()
        self.vae.eval()  
        self.noise_scheduler.timesteps = torch.linspace(0, 999, 1000) 

        val_loss = 0
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                text_inputs = self.tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                text_embeds_1 = self.text_encoder_1(text_inputs.input_ids)[0]
                pooled_embeds_1 = self.text_encoder_1(text_inputs.input_ids)[1]
                text_embeds_2 = self.text_encoder_2(text_inputs.input_ids)[0]
                pooled_embeds_2 = self.text_encoder_2(text_inputs.input_ids)[1]
                
                text_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
                pooled_embeds = torch.cat([pooled_embeds_1, pooled_embeds_2], dim=-1)
                
                add_time_ids = self._get_add_time_ids(batch, text_embeds.dtype)
                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                
                # Process image
                latents = self.vae.encode(batch["image"].to(self.device)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],), device=self.device)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        
        val_loss /= len(self.test_dataloader)
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        
    def generate_samples(self, num_samples: int = 2):
        cross_attention_kwargs = {"scale": 1.0}
        if hasattr(self.unet, "lora_state_dict"):
            cross_attention_kwargs["lora_scale"] = self.unet.lora_alpha

        sample_dir = self.config.paths['output'] / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        self.unet.eval()
        style_config = self.config.get_training_config()['style']
        
        for i in range(num_samples):
            sample = next(iter(self.test_dataloader))
            
            text_inputs = self.tokenizer(
                sample['prompt'],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Use pipeline for convenience
            images = self.model(
                prompt=sample['prompt'][0],
                negative_prompt=style_config['negative_prompt'],
                num_inference_steps=30,
                guidance_scale=7.5,
                cross_attention_kwargs=cross_attention_kwargs
            ).images
            
            images[0].save(sample_dir / f"sample_{i}.png")

def main():
    config = ProjectConfig()
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()