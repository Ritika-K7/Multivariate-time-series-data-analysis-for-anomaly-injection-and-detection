import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
script_path = Path(__file__).resolve()
sys.path.append(str(script_path.parent.parent))

from models.autoencoder import Autoencoder
from utils.sliding_window import create_windows
from utils.normalization import normalize

WINDOW_SIZE = 50
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_SENSORS = 3


def train(model, train_data, epochs=30, batch_size=64, lr=1e-3):
    """Train autoencoder on normal data."""
    loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            x_flat = x.view(x.size(0), -1)
            
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


def compute_error(model, windows):
    """Compute reconstruction errors for windows."""
    model.eval()
    with torch.no_grad():
        recon = model(windows)
        errors = torch.mean(
            (recon - windows.view(windows.size(0), -1)) ** 2,
            dim=1
        )
    return errors.numpy()


if __name__ == "__main__":
    print("Training autoencoder on NORMAL data only...")
    
    # Get project root directory (parent of training/)
    project_root = Path(__file__).resolve().parent.parent
    
    # Load normal data
    normal_file = project_root / "results" / "results_normal_data_gen" / "normal_data.csv"
    df = pd.read_csv(normal_file)
    data = df[['s1', 's2', 's3']].values
    print(f"Loaded data: {data.shape}")
    
    # Create windows
    windows = create_windows(data, WINDOW_SIZE)
    print(f"Created {len(windows)} windows")
    
    # Normalize
    windows_flat = windows.reshape(-1, NUM_SENSORS)
    windows_norm, mean, std = normalize(windows_flat)
    windows_norm = windows_norm.reshape(len(windows), WINDOW_SIZE, NUM_SENSORS)
    
    # Convert to tensors
    train_data = torch.FloatTensor(windows_norm)
    
    # Train model
    input_dim = WINDOW_SIZE * NUM_SENSORS
    model = Autoencoder(input_dim)
    train(model, train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
    
    # Compute training errors and threshold
    train_errors = compute_error(model, train_data)
    threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    
    print(f"\nTraining errors - Mean: {np.mean(train_errors):.6f}, Std: {np.std(train_errors):.6f}")
    print(f"Anomaly threshold (mean + 3*std): {threshold:.6f}")
    
    # Save everything
    results_dir = project_root / "results" / "results_autoencoder_training"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = results_dir / "autoencoder_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'window_size': WINDOW_SIZE
    }, model_path)
    
    # Save normalization params
    norm_params_path = results_dir / "normalization_params.json"
    with open(norm_params_path, 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
    
    # Save threshold
    threshold_path = results_dir / "thresholds.json"
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)
    
    print(f"\n Saved to: {results_dir}")
