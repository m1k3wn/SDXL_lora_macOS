import logging
import json
from pathlib import Path
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
        
        # Construct prompt with trigger word
        prompt = f"{self.config.trigger_word}, {item['caption']}"
        
        return {
            'image': image,
            'prompt': prompt
        }

class LoRATrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_accelerator()
        self.load_model()
        self.setup_datasets()
        
    def setup_accelerator(self):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get_training_config()['training']['gradient_accumulation_steps'],
            mixed_precision="fp16"
        )
        
    def load_model(self):
        self.model = StableDiffusionXLPipeline.from_pretrained(
            self.config.paths['base_model'],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config
        )
        
    def setup_datasets(self):
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
        """Configure LoRA layers in the model"""
        from diffusers.loaders import AttnProcsLayers
        from diffusers.models.attention_processor import LoRAAttnProcessor

        lora_config = self.config.get_training_config()['lora']
        
        # Add LoRA layers to attention processors
        attn_procs = {}
        for name in self.model.attn_processors.keys():
            attn_procs[name] = LoRAAttnProcessor(
                hidden_size=self.model.config.hidden_size,
                cross_attention_dim=self.model.config.cross_attention_dim,
                rank=lora_config['rank'],
                network_alpha=lora_config['alpha']
            )
        
        self.model.set_attn_processor(attn_procs)
        
    def train(self):
        self.logger.info("Starting training...")
        training_config = self.config.get_training_config()['training']
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate']
        )
        
        # Prepare model and optimizer
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader
        )
        
        global_step = 0
        for epoch in range(training_config['num_epochs']):
            self.model.train()
            progress_bar = tqdm(total=len(train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.model):
                    # Get model output
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True
                    )
                    
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    optimizer.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
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
        self.accelerator.save(self.model.state_dict(), save_path)
        
    def validate(self):
        """Run validation on test dataset"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True
                )
                val_loss += outputs.loss.item()
                
        val_loss /= len(self.test_dataloader)
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Generate sample images
        self.generate_samples()
        
    def generate_samples(self, num_samples: int = 2):
        """Generate sample images during validation"""
        sample_dir = self.config.paths['output'] / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            sample = next(iter(self.test_dataloader))
            image = self.model(
                prompt=sample['prompt'],
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
            image.save(sample_dir / f"sample_{i}.png")

def main():
    config = ProjectConfig()
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()