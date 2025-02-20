from diffusers import StableDiffusionXLPipeline
import torch

# Convert and save in Diffusers format
pipe = StableDiffusionXLPipeline.from_single_file(
    "/Volumes/Desktop SSD/AI-Projects/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Save in Diffusers format using your project directory
pipe.save_pretrained("/Volumes/Desktop SSD/AI-Projects/SDXL_LORAS/base_model")

print("Model converted and saved successfully!")