import torch
import numpy as np

def compute_error(model, windows):
    model.eval()
    with torch.no_grad():
        windows_flat = windows.view(windows.size(0), -1)
        recon = model(windows)
        errors = torch.mean((recon - windows_flat) ** 2, dim=1)
    return errors.cpu().numpy()
