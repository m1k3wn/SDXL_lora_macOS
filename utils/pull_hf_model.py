from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv("HF_TOKEN")
print(f"Token loaded: {token is not None}")

from diffusers import StableDiffusionXLPipeline
import torch
print("Libraries imported")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    token=token,  # Add token here
    torch_dtype=torch.float32,
    use_safetensors=True
)
print("Pipeline created")

pipe.save_pretrained("./base_model")
print("Model saved")