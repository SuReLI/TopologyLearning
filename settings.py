import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Prevent problems with cuda version as we don't need cuda in this project.