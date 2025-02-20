# config.py

import os
from pathlib import Path

# Project Settings
PROJECT_NAME = "WOH_lora"
TRIGGER_WORD = "WOH portrait style"

class ProjectConfig:
    def __init__(self):
        # Base paths
        self.base_path = Path("/Volumes/Desktop SSD/AI-Projects/SDXL_LORAS")
        
        # Project specific paths
        self.project_name = PROJECT_NAME
        self.trigger_word = TRIGGER_WORD
        self.project_dir = self.base_path / self.project_name
        
        # Create project directory if it doesn't exist
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all project paths
        self.paths = {
            "base_model": self.base_path / "base_model",
            "dataset": {
                "test": self.project_dir / "dataset/test_dataset",    # 512px images
                "train": self.project_dir / "dataset/train_dataset",  # 1024px images
            },
            "metadata": {
                "test": self.project_dir / "dataset/test_metadata.json",
                "train": self.project_dir / "dataset/train_metadata.json"
            },
            "output": self.project_dir / "output",
            "config": self.project_dir / "config.yaml"
        }
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories in the project structure"""
        for path in self.paths.values():
            if isinstance(path, dict):
                for subpath in path.values():
                    if not subpath.suffix:  # Only create directory if it's not a file path
                        subpath.mkdir(parents=True, exist_ok=True)
            else:
                if not path.suffix:  # Only create directory if it's not a file path
                    path.mkdir(parents=True, exist_ok=True)
    
    def get_training_config(self):
        """Get training-specific configuration"""
        return {
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 1,
                "num_epochs": 100,
                "gradient_accumulation_steps": 4,
                "seed": 42,
                "mixed_precision": "fp16",
                "save_steps": 100,
                "validation_steps": 50
            },
            "lora": {
                "rank": 4,
                "alpha": 4,
                "dropout": 0.0
            },
            "style": {
                "trigger_word": self.trigger_word,
                "base_prompt": "{trigger_word}, monochromatic expressive portrait artwork, dramatic black ink marks, dynamic linework, bold shadows, gestural marks, textural smudges, greyscale shading, rough marks, dramatic contrast",
                "negative_prompt": "photorealistic, photograph, color, watercolor, 3d render, anime, cartoon"
            }
        }
    
    def __str__(self):
        """Pretty print the configuration"""
        return f"""Project Configuration:
Project Name: {self.project_name}
Trigger Word: {self.trigger_word}
Base Path: {self.base_path}
Dataset Paths:
  Test Dataset (512px): {self.paths['dataset']['test']}
  Train Dataset (1024px): {self.paths['dataset']['train']}
Metadata:
  Test: {self.paths['metadata']['test']}
  Train: {self.paths['metadata']['train']}
Output: {self.paths['output']}
"""

if __name__ == "__main__":
    # Test configuration
    config = ProjectConfig()
    print(config)