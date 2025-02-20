import logging
from pathlib import Path
import sys
import argparse
from config import ProjectConfig
from prepare_data import DataPreparator
from train import LoRATrainer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train LoRA for SDXL')
    parser.add_argument('--mode', choices=['test', 'production'], default='production', help='Training mode (test=5 epochs, production=100 epochs)')
    parser.add_argument('--force-resize', action='store_true', help='Force image resizing')
    parser.add_argument('--skip-validation', action='store_true', help='Skip dataset validation')
    args = parser.parse_args()
    
    logger = setup_logging()
    config = ProjectConfig(mode=args.mode)

    
    logger.info(f"Starting LoRA training for project: {config.project_name}")
    logger.info(f"Trigger word: {config.trigger_word}")
    
    if not args.skip_validation:
        preparator = DataPreparator(config)
        if not preparator.prepare_dataset(force_resize=args.force_resize):
            logger.error("Dataset preparation failed")
            sys.exit(1)
    
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()