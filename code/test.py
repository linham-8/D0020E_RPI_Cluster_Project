import torch

result = torch.tensor([6.0, 7.0])

print(f"Shape: {result.shape}")
print(f"First Item: {result[0].item()}")
print(f"Second Item: {result[1].item()}")
print(f"Device: {result.device}") 
