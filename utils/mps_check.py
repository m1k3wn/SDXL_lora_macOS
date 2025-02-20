import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Current device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")