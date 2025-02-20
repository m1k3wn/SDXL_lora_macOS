import os
from pathlib import Path
import json
import shutil
from PIL import Image
import logging
from typing import Dict, List, Tuple
from config import ProjectConfig

class DataPreparator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_metadata(self) -> Tuple[bool, Dict]:
        """Validate metadata files and return status and stats"""
        stats = {"test": 0, "train": 0, "errors": []}
        
        for dataset_type in ["test", "train"]:
            metadata_path = self.config.paths['metadata'][dataset_type]
            if not metadata_path.exists():
                stats["errors"].append(f"Missing {dataset_type} metadata file")
                continue
                
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if "files" not in metadata:
                    stats["errors"].append(f"Invalid {dataset_type} metadata structure")
                    continue
                    
                stats[dataset_type] = len(metadata["files"])
                
                # Validate each file exists and has caption
                for item in metadata["files"]:
                    if "filename" not in item or "caption" not in item:
                        stats["errors"].append(f"Missing filename or caption in {dataset_type} metadata")
                        continue
                        
                    image_path = self.config.paths['dataset'][dataset_type] / item["filename"]
                    if not image_path.exists():
                        stats["errors"].append(f"Missing image file: {item['filename']}")
                        
            except json.JSONDecodeError:
                stats["errors"].append(f"Invalid JSON in {dataset_type} metadata")
                
        return len(stats["errors"]) == 0, stats
    
    def validate_images(self) -> Tuple[bool, Dict]:
        """Validate image sizes and formats"""
        stats = {"test": {"valid": 0, "invalid": 0}, 
                "train": {"valid": 0, "invalid": 0},
                "errors": []}
                
        size_requirements = {
            "test": (512, 512),
            "train": (1024, 1024)
        }
        
        for dataset_type in ["test", "train"]:
            dataset_path = self.config.paths['dataset'][dataset_type]
            required_size = size_requirements[dataset_type]
            
            for image_path in dataset_path.glob("*.png"):
                try:
                    with Image.open(image_path) as img:
                        if img.size != required_size:
                            stats[dataset_type]["invalid"] += 1
                            stats["errors"].append(
                                f"{image_path.name} has incorrect size: {img.size} (should be {required_size})"
                            )
                        else:
                            stats[dataset_type]["valid"] += 1
                except Exception as e:
                    stats[dataset_type]["invalid"] += 1
                    stats["errors"].append(f"Error processing {image_path.name}: {str(e)}")
                    
        return len(stats["errors"]) == 0, stats
    
    def resize_images(self, force: bool = False) -> Dict:
        """Resize images to required dimensions"""
        stats = {"resized": 0, "skipped": 0, "errors": []}
        
        size_requirements = {
            "test": (512, 512),
            "train": (1024, 1024)
        }
        
        for dataset_type, required_size in size_requirements.items():
            dataset_path = self.config.paths['dataset'][dataset_type]
            
            for image_path in dataset_path.glob("*.png"):
                try:
                    with Image.open(image_path) as img:
                        if img.size != required_size or force:
                            resized = img.resize(required_size, Image.Resampling.LANCZOS)
                            resized.save(image_path)
                            stats["resized"] += 1
                        else:
                            stats["skipped"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error resizing {image_path.name}: {str(e)}")
                    
        return stats
    
    def prepare_dataset(self, force_resize: bool = False) -> bool:
        """Main method to prepare and validate the dataset"""
        self.logger.info("Starting dataset preparation...")
        
        # Validate metadata
        metadata_valid, metadata_stats = self.validate_metadata()
        if not metadata_valid:
            self.logger.error("Metadata validation failed:")
            for error in metadata_stats["errors"]:
                self.logger.error(f"  - {error}")
            return False
            
        self.logger.info(f"Found {metadata_stats['test']} test and {metadata_stats['train']} train images in metadata")
        
        # Resize images if needed
        resize_stats = self.resize_images(force=force_resize)
        if resize_stats["errors"]:
            self.logger.error("Some images failed to resize:")
            for error in resize_stats["errors"]:
                self.logger.error(f"  - {error}")
            return False
            
        self.logger.info(f"Resized {resize_stats['resized']} images, skipped {resize_stats['skipped']}")
        
        # Final validation
        images_valid, image_stats = self.validate_images()
        if not images_valid:
            self.logger.error("Image validation failed:")
            for error in image_stats["errors"]:
                self.logger.error(f"  - {error}")
            return False
            
        self.logger.info("Dataset preparation completed successfully")
        return True

def main():
    config = ProjectConfig()
    preparator = DataPreparator(config)
    success = preparator.prepare_dataset()
    if not success:
        print("Dataset preparation failed. Please check the errors above.")
        exit(1)

if __name__ == "__main__":
    main()