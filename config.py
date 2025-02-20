import os
from pathlib import Path

# Project Settings
PROJECT_NAME = "WOH_lora"
TRIGGER_WORD = "WOH portrait style"

# LoRA Parameters
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
SEED = 42

# For testing 
LEARNING_RATE_TEST = 2e-4
EPOCHS_TEST = 5
SAVE_STEPS_TEST = 50
VALIDATION_STEPS_TEST = 25

# For production
LEARNING_RATE_PRODUCTION = 1e-4
EPOCHS_PRODUCTION = 100
MIXED_PRECISION = "fp16"
SAVE_STEPS_PRODUCTION = 100
VALIDATION_STEPS_PRODUCTION = 50

# Style Settings
BASE_PROMPT = "{trigger_word}, monochromatic expressive portrait artwork, dramatic black ink marks, dynamic linework, bold shadows, gestural marks, textural smudges, greyscale shading, rough marks, dramatic contrast"
NEGATIVE_PROMPT = "photorealistic, photograph, color, watercolor, 3d render, anime, cartoon"

class ProjectConfig:
    def __init__(self, mode="production"):
        self.mode = mode
        self.base_path = Path("/Volumes/Desktop SSD/AI-Projects/SDXL_LORAS")
        self.project_name = PROJECT_NAME
        self.trigger_word = TRIGGER_WORD
        self.project_dir = self.base_path / self.project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.paths = self._setup_paths()
        self._create_directories()
    
    def _setup_paths(self):
        return {
            "base_model": Path("/base_model"),
            "dataset": {
                "test": self.project_dir / "dataset/test_dataset",
                "train": self.project_dir / "dataset/train_dataset",
            },
            "metadata": {
                "test": self.project_dir / "dataset/test_metadata.json",
                "train": self.project_dir / "dataset/train_metadata.json"
            },
            "output": self.project_dir / "output",
            "config": self.project_dir / "config.yaml"
        }
    
    def _create_directories(self):
        for path in self.paths.values():
            if isinstance(path, dict):
                for subpath in path.values():
                    if not subpath.suffix:
                        subpath.mkdir(parents=True, exist_ok=True)
            elif not path.suffix:
                path.mkdir(parents=True, exist_ok=True)
    
    def get_training_config(self):
        base_config = {
            "training": {
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "seed": SEED,
                "mixed_precision": MIXED_PRECISION,
            },
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT
            },
            "style": {
                "trigger_word": self.trigger_word,
                "base_prompt": BASE_PROMPT,
                "negative_prompt": NEGATIVE_PROMPT
            }
        }

        if self.mode == "test":
            base_config["training"].update({
                "learning_rate": LEARNING_RATE_TEST,
                "num_epochs": EPOCHS_TEST,
                "save_steps": SAVE_STEPS_TEST,
                "validation_steps": VALIDATION_STEPS_TEST
            })
        else:
            base_config["training"].update({
                "learning_rate": LEARNING_RATE_PRODUCTION,
                "num_epochs": EPOCHS_PRODUCTION,
                "save_steps": SAVE_STEPS_PRODUCTION,
                "validation_steps": VALIDATION_STEPS_PRODUCTION
            })

        return base_config

    def __str__(self):
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