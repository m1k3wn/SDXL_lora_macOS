import os
import json
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class CustomArtDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_paths = list(self.data_dir.glob("*.png"))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption_path = image_path.with_suffix('.txt')
        
        image = Image.open(image_path).convert('RGB')
        
        with open(caption_path, 'r') as f:
            caption = f.read().strip()
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": self.process_image(image),
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0]
        }
    
    def process_image(self, image):
        image = torchvision.transforms.functional.to_tensor(image)
        image = 2.0 * image - 1.0
        return image

def main():
    # Load configuration
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['dataset']['gradient_accumulation_steps'],
        mixed_precision="fp16"
    )
    
    # Load SDXL pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config['pretrained_model_name_or_path'],
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Setup LoRA configuration
    lora_config = config['lora_config']
    
    # Initialize LoRA layers
    for name, module in pipeline.text_encoder.named_modules():
        if any(target in name for target in lora_config['target_modules']):
            module = setup_lora_layer(
                module,
                lora_config['r'],
                lora_config['lora_alpha'],
                lora_config['bias']
            )
    
    # Setup dataset and dataloader
    dataset = CustomArtDataset(
        config['dataset']['train_data_dir'],
        pipeline.tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['train_batch_size'],
        shuffle=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        get_lora_parameters(pipeline),
        lr=config['optimizer']['learning_rate'],
        betas=(config['optimizer']['adam_beta1'], config['optimizer']['adam_beta2']),
        weight_decay=config['optimizer']['adam_weight_decay'],
        eps=config['optimizer']['adam_epsilon']
    )
    
    # Training loop
    for epoch in range(config['dataset']['num_train_epochs']):
        train_one_epoch(
            pipeline,
            dataloader,
            optimizer,
            accelerator,
            epoch,
            config
        )
        
        # Save checkpoint
        if (epoch + 1) % config['dataset']['save_epochs'] == 0:
            save_lora_checkpoint(
                pipeline,
                config['output_dir'],
                f"checkpoint-{epoch+1}"
            )

if __name__ == "__main__":
    main()