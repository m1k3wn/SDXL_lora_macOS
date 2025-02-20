import logging
import json
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import safetensors
from tqdm.auto import tqdm
from config import ProjectConfig

class LoRADataset(Dataset):
    def __init__(self, config: ProjectConfig, dataset_type: str):
        self.config = config
        self.dataset_type = dataset_type
        self.image_dir = config.paths['dataset'][dataset_type]
        self.metadata_path = config.paths['metadata'][dataset_type]
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.metadata['files'])
    
    def __getitem__(self, idx):
        item = self.metadata['files'][idx]
        image_path = self.image_dir / item['filename']
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Construct prompt with trigger word and style
        style_config = self.config.get_training_config()['style']
        base_prompt = style_config['base_prompt'].format(trigger_word=style_config['trigger_word'])
        prompt = f"{item['caption']}, {base_prompt}"
        
        return {
            'image': image,
            'prompt': prompt,
            'negative_prompt': style_config['negative_prompt']
        }

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
        
        # Move models to device
        self.model.to(self.device)
        
    def load_model(self):
        self.model = StableDiffusionXLPipeline.from_pretrained(
            self.config.paths['base_model'],
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config
        )
        
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
        
    def setup_lora(self):
        """Configure LoRA layers for SDXL"""
        lora_config = self.config.get_training_config()['lora']
        
        # Load UNet with LoRA config
        self.model.unet = UNet2DConditionModel.from_pretrained(
            self.config.paths['base_model'] / "unet",
            torch_dtype=torch.float32,
            use_lora=True,
            lora_r=lora_config['rank']
        )
        self.model.unet.train()
            
    def train(self):
        self.logger.info("Starting training...")
        training_config = self.config.get_training_config()['training']
        
        # Setup LoRA
        self.setup_lora()
        
        optimizer = torch.optim.AdamW(
            self.model.unet.parameters(),
            lr=training_config['learning_rate']
        )
        
        # Prepare model and optimizer
        self.model.unet, optimizer, train_dataloader = self.accelerator.prepare(
            self.model.unet, optimizer, self.train_dataloader
        )
        
        global_step = 0
        for epoch in range(training_config['num_epochs']):
            self.model.unet.train()
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model.unet):

                    # Get conditioning input
                    prompt_ids = self.tokenizer(
                        batch["prompt"], 
                        padding="max_length",
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).input_ids.to(self.device)

                    with torch.no_grad():
                        text_embeds_1 = self.text_encoder_1(prompt_ids).last_hidden_state  # [batch, 77, 1280]
                        pooled_text_embeds_1 = self.text_encoder_1(prompt_ids).pooler_output  # [batch, 1280]
                        text_embeds_2 = self.text_encoder_2(prompt_ids).last_hidden_state  # [batch, 77, 768]
                        pooled_text_embeds_2 = self.text_encoder_2(prompt_ids).pooler_output  # [batch, 1280]

                    # Fix: Concatenate everything to match SDXL expectations
                    text_embeds = torch.cat([
                        text_embeds_1,  # [batch, 77, 1280]
                        text_embeds_2,  # [batch, 77, 768]
                        pooled_text_embeds_1.unsqueeze(1).repeat(1, 77, 1),  # Broadcast to [batch, 77, 1280]
                    ], dim=-1)  # Final shape: [batch, 77, 2816]


                    # Concatenate all embeddings for SDXL
                    text_embeds = torch.cat([
                        text_embeds_1,  # [batch, 77, 1280]
                        text_embeds_2,  # [batch, 77, 1280]
                    ], dim=-1)  # Result: [batch, 77, 2560]

                    # Create micro condition for SDXL
                    micro_cond = torch.cat([
                        pooled_text_embeds_1.unsqueeze(1),  # [batch, 1, 1280]
                    ], dim=1)  # [batch, 1, 1280]

                    # Create time embeddings
                    original_size = (1024, 1024)
                    target_size = (1024, 1024)
                    crops_coords_top_left = (0, 0)
                    aspect_ratio = target_size[0] / target_size[1]

                    add_time_ids = torch.tensor([
                        1.0,  # aspect_ratio (dummy value)
                        1024, 1024,  # original_size
                        0, 0,  # crops_coords_top_left
                        1.0  # target_size / original_size (dummy value)
                    ], device=self.device, dtype=text_embeds.dtype).unsqueeze(0).unsqueeze(1)  # [1, 1, 6]

                    # Expand to match `text_embeds` shape
                    add_time_ids = add_time_ids.expand(text_embeds.shape[0], 77, -1)  # [batch, 77, 6]

                    # Set added conditioning kwargs with correct shapes
                    added_cond_kwargs = {
                        "text_embeds": text_embeds,  # [batch, 77, 2816]
                        "time_ids": add_time_ids  # [batch, 77, 6]
                    }

                    # Debug prints to verify shapes
                    print(f"text_embeds shape: {text_embeds.shape}")  # Should be [batch, 77, 2560]
                    print(f"add_time_ids shape: {add_time_ids.shape}")  # Should be [batch, 6]
                    print(f"micro_cond shape: {micro_cond.shape}")  # Should be [batch, 1, 1280]

                    # Convert images to latents using VAE
                    with torch.no_grad():
                        latents = self.model.vae.encode(batch["image"].to(self.device)).latent_dist.sample()

                    # Scale latents for SDXL
                    latents = latents * self.model.vae.config.scaling_factor  

                    # Generate random noise
                    noise = torch.randn_like(latents)

                    # Select random timesteps
                    timesteps = torch.randint(0, self.model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device)

                    # Add noise to latents
                    noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)

                    # Forward pass through UNet
                    noise_pred = self.model.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,  # [batch, 77, 2560]
                        added_cond_kwargs=added_cond_kwargs
                    ).sample

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backpropagation
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                global_step += 1
                
                if global_step % training_config['save_steps'] == 0:
                    self.save_model(f"checkpoint-{global_step}")
                    
                if global_step % training_config['validation_steps'] == 0:
                    self.validate()
            
            progress_bar.close()
            
        # Final save
        self.save_model("final")
        
    def save_model(self, tag: str):
        """Save LoRA weights"""
        save_path = self.config.paths['output'] / f"lora_{tag}.safetensors"
        
        # Get unwrapped model
        unwrapped_unet = self.accelerator.unwrap_model(self.model.unet)
        state_dict = unwrapped_unet.state_dict()
        
        # Filter for LoRA weights only
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
        self.accelerator.save(lora_state_dict, save_path)
        
    def validate(self):
        self.model.unet.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                prompt_ids = self.tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).input_ids.to(self.model.device)
                
                # Forward pass through UNet
                latents = self.model.vae.encode(batch["image"].to(self.model.device)).latent_dist.sample()
                latents = latents * self.model.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)
                
                # Get UNet predictions
                # Generate time conditioning for validation
                add_time_ids = torch.tensor([
                    1.0,  # aspect_ratio (dummy value)
                    1024, 1024,  # original_size
                    0, 0,  # crops_coords_top_left
                    1.0  # target_size / original_size (dummy value)
                ], device=self.device, dtype=torch.float32).unsqueeze(0)  # [1, 6]

                # Ensure text embeddings
                with torch.no_grad():
                    text_embeds_1 = self.text_encoder_1(prompt_ids).last_hidden_state
                    text_embeds_2 = self.text_encoder_2(prompt_ids).last_hidden_state
                text_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)  # [batch, 77, 2816]

                # Add missing added_cond_kwargs
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": add_time_ids.expand(text_embeds.shape[0], -1)
                }

                # Forward pass through UNet
                noise_pred = self.model.unet(
                    noisy_latents, timesteps, prompt_ids, added_cond_kwargs=added_cond_kwargs
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_loss += loss.item()
                
        val_loss /= len(self.test_dataloader)
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        self.generate_samples()
        
    def generate_samples(self, num_samples: int = 2):
        sample_dir = self.config.paths['output'] / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        style_config = self.config.get_training_config()['style']
        
        # Set to eval mode for inference
        self.model.unet.eval()
        
        for i in range(num_samples):
            sample = next(iter(self.test_dataloader))
            
            with torch.no_grad():
                images = self.model(
                    prompt=sample['prompt'],
                    negative_prompt=sample['negative_prompt'],
                    num_inference_steps=40,
                    guidance_scale=9.0,
                    height=1024,
                    width=1024
                ).images
                
                images[0].save(sample_dir / f"sample_{i}.png")

def main():
    config = ProjectConfig()
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()