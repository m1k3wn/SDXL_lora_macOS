from config import TrainingConfig
from prepare_data import DataPreparator
from train import LoRATrainer

def main():
    # Initialize with test or production config
    config = TrainingConfig(mode="test")  # or "production"
    
    # Prepare dataset
    preparator = DataPreparator(config)
    preparator.prepare_dataset()
    
    # Train model
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()