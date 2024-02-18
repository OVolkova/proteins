import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
import os

print(os.cpu_count())

print(help(torch.utils.data.DataLoader))
