import torch

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
