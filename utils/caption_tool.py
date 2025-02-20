import sys
from pathlib import Path

# Add parent directory to Python path so we can import config.py
sys.path.append(str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
from config import ProjectConfig

class GridCaptionUI:
    def __init__(self, root, config, dataset_type="test"):
        self.root = root
        self.root.title(f"LoRA Training Caption Tool - {config.project_name}")
        
        self.config = config
        self.dataset_type = dataset_type
        self.image_dir = config.paths['dataset'][dataset_type]
        self.metadata_file = config.paths['metadata'][dataset_type]
        
        self.metadata = self.load_metadata()
        self.image_files = self.get_image_files()
        self.caption_entries = {}
        
        if not self.image_files:
            messagebox.showerror("Error", f"No PNG images found in {self.image_dir}")
            root.destroy()
            return
        
        self.setup_ui()
    
    def get_image_files(self):
        """Get all PNG files from the image directory"""
        return list(self.image_dir.glob("*.png"))
    
    def get_base_filename(self, filename):
        """Convert filename to base name for matching between datasets
        WOH_lora_17SML.png -> WOH_lora_17
        WOH_lora_17.png -> WOH_lora_17
        """
        base = filename.rsplit('.', 1)[0]  # Remove extension
        if base.endswith('SML'):
            base = base[:-3]  # Remove 'SML'
        return base

    def load_metadata(self):
        """Load existing metadata or create new, copying from test dataset if available"""
        # If this is the training dataset, try to load test metadata first
        if self.dataset_type == "train":
            test_metadata_file = self.config.paths['metadata']['test']
            if test_metadata_file.exists():
                print(f"Found test metadata, importing captions...")
                with open(test_metadata_file, 'r') as f:
                    test_metadata = json.load(f)
                    
                    # Create a mapping of base filenames to captions
                    caption_map = {
                        self.get_base_filename(item["filename"]): item["caption"]
                        for item in test_metadata["files"]
                    }
                    
                    # Map captions to training filenames
                    mapped_metadata = {
                        "files": [],
                        "dataset_type": "train"
                    }
                    
                    for train_file in self.get_image_files():
                        base_name = self.get_base_filename(train_file.name)
                        if base_name in caption_map:
                            mapped_metadata["files"].append({
                                "filename": train_file.name,
                                "caption": caption_map[base_name]
                            })
                            print(f"Mapped caption from {base_name}SML.png to {train_file.name}")
                        
                    return mapped_metadata
        
        # Otherwise load this dataset's metadata if it exists
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        
        return {"files": [], "dataset_type": self.dataset_type}
    
    def setup_ui(self):
        # Dataset type indicator
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        dataset_label = ttk.Label(
            header_frame, 
            text=f"Dataset: {self.dataset_type.upper()} ({self.image_dir})",
            font=('TkDefaultFont', 10, 'bold')
        )
        dataset_label.pack(side='left')
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid frame inside canvas
        self.grid_frame = ttk.Frame(canvas)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas
        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        
        # Populate grid with images and caption fields
        for idx, img_path in enumerate(self.image_files):
            frame = ttk.Frame(self.grid_frame)
            frame.grid(row=idx, column=0, pady=10, sticky="w")
            
            # Load and display image
            image = Image.open(img_path)
            image.thumbnail((200, 200))  # Smaller thumbnails for grid view
            photo = ImageTk.PhotoImage(image)
            img_label = ttk.Label(frame, image=photo)
            img_label.image = photo
            img_label.grid(row=0, column=0, padx=5)
            
            # Caption input area
            caption_frame = ttk.Frame(frame)
            caption_frame.grid(row=0, column=1, padx=5, sticky="n")
            
            filename = img_path.name
            ttk.Label(caption_frame, text=filename).pack(anchor='w')
            
            # Get existing caption if any
            existing_caption = ""
            for item in self.metadata["files"]:
                if item["filename"] == filename:
                    existing_caption = item["caption"]
                    break
            
            caption_entry = ttk.Entry(caption_frame, width=50)
            caption_entry.insert(0, existing_caption)
            caption_entry.pack(pady=5)
            
            self.caption_entries[filename] = caption_entry
        
        # Button frame at the bottom
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame, 
            text="Save All Captions", 
            command=self.save_all
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Switch Dataset", 
            command=self.switch_dataset
        ).pack(side='left', padx=5)
        
        # Configure canvas scrolling
        self.grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    
    def switch_dataset(self):
        """Switch between test and train datasets"""
        if self.save_all(silent=True):  # Save current work first
            new_type = "train" if self.dataset_type == "test" else "test"
            self.root.destroy()  # Close current window
            
            # Open new window with other dataset
            root = tk.Tk()
            app = GridCaptionUI(root, self.config, new_type)
            root.mainloop()
    
    def save_all(self, silent=False):
        """Save all captions to metadata file"""
        try:
            self.metadata["files"] = []
            for filename, entry in self.caption_entries.items():
                caption = entry.get().strip()
                if caption:  # Only save if caption is not empty
                    self.metadata["files"].append({
                        "filename": filename,
                        "caption": caption
                    })
            
            # Save to file
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            if not silent:
                messagebox.showinfo(
                    "Success", 
                    f"Saved {len(self.metadata['files'])} captions to metadata file"
                )
            return True
            
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to save captions: {str(e)}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Grid Caption Tool for LoRA Training Images')
    parser.add_argument('--dataset', type=str, choices=['test', 'train'], 
                       default='test', help='Dataset to caption (default: test)')
    args = parser.parse_args()
    
    config = ProjectConfig()
    
    # Print project info
    print(config)
    
    root = tk.Tk()
    app = GridCaptionUI(root, config, args.dataset)
    root.mainloop()

if __name__ == "__main__":
    main()