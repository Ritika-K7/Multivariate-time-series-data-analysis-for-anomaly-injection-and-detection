import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt

# Add parent directory to path for imports (must be before importing models/utils)
script_path = Path(__file__).resolve()
sys.path.append(str(script_path.parent.parent))

from models.autoencoder import Autoencoder
from utils.sliding_window import create_windows

WINDOW_SIZE = 50
NUM_SENSORS = 3


def compute_error(model, windows):
    """Compute reconstruction errors for windows."""
    model.eval()
    with torch.no_grad():
        windows_flat = windows.view(windows.size(0), -1)
        recon = model(windows)
        errors = torch.mean((recon - windows_flat) ** 2, dim=1)
    return errors.cpu().numpy()


def normalize_with_params(data, mean, std):
    """Normalize data using pre-computed mean and std."""
    return (data - mean) / std


if __name__ == "__main__":
    print("Testing autoencoder on anomaly data...")
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Step 1: Load trained model
    model_dir = project_root / "results" / "results_autoencoder_training"
    checkpoint = torch.load(model_dir / "autoencoder_model.pt")
    
    model = Autoencoder(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model (input_dim={checkpoint['input_dim']}, window_size={checkpoint['window_size']})")
    
    # Load normalization parameters
    with open(model_dir / "normalization_params.json", 'r') as f:
        norm_params = json.load(f)
    train_mean = np.array(norm_params['mean'])
    train_std = np.array(norm_params['std'])
    print(f"Loaded normalization params")
    
    # Load threshold
    with open(model_dir / "thresholds.json", 'r') as f:
        thresholds = json.load(f)
    threshold = thresholds['threshold']
    print(f"Loaded threshold: {threshold:.6f}")
    
    # Step 2: Load synthetic anomaly-injected data
    anomaly_file = project_root / "results" / "results_anomaly_data_generation" / "mixed_anomaly.csv"
    df = pd.read_csv(anomaly_file)
    data = df[['s1', 's2', 's3']].values
    true_labels = df['label'].values
    print(f"Loaded anomaly data: {data.shape}")
    print(f"   - Total samples: {len(data)}")
    print(f"   - Anomalies in data: {int(true_labels.sum())}")
    
    # Step 3: Create sliding windows
    windows = create_windows(data, WINDOW_SIZE)
    print(f"Created {len(windows)} windows")
    
    # Step 4: Normalize using training mean/std
    windows_flat = windows.reshape(-1, NUM_SENSORS)
    windows_norm = normalize_with_params(windows_flat, train_mean, train_std)
    windows_norm = windows_norm.reshape(len(windows), WINDOW_SIZE, NUM_SENSORS)
    
    # Convert to tensors
    test_data = torch.FloatTensor(windows_norm)
    
    # Step 5: Compute reconstruction errors
    print("\nComputing reconstruction errors...")
    errors = compute_error(model, test_data)
    print(f"Computed errors for {len(errors)} windows")
    print(f"   - Mean error: {np.mean(errors):.6f}")
    print(f"   - Max error: {np.max(errors):.6f}")
    print(f"   - Min error: {np.min(errors):.6f}")
    
    # Step 6: Compare errors with threshold
    predicted_labels = (errors > threshold).astype(int)
    
    # Step 7: Flag anomalies
    # Map window-level predictions back to time-step level
    # Each window i covers time steps [i, i+WINDOW_SIZE)
    time_step_predictions = np.zeros(len(data))
    time_step_errors = np.zeros(len(data))
    time_step_counts = np.zeros(len(data))
    
    for i, (pred, error) in enumerate(zip(predicted_labels, errors)):
        start_idx = i
        end_idx = min(i + WINDOW_SIZE, len(data))
        if pred == 1:
            time_step_predictions[start_idx:end_idx] = 1
        # Accumulate errors (will average later)
        time_step_errors[start_idx:end_idx] += error
        time_step_counts[start_idx:end_idx] += 1
    
    # Average errors for time steps covered by multiple windows
    time_step_errors = time_step_errors / np.maximum(time_step_counts, 1)
    
    # Calculate metrics
    # For comparison, we'll use the true labels (excluding the last WINDOW_SIZE-1 points that don't have windows)
    comparison_labels = true_labels[:len(time_step_predictions)]
    
    tp = np.sum((time_step_predictions == 1) & (comparison_labels == 1))
    fp = np.sum((time_step_predictions == 1) & (comparison_labels == 0))
    fn = np.sum((time_step_predictions == 0) & (comparison_labels == 1))
    tn = np.sum((time_step_predictions == 0) & (comparison_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\nDetection Results:")
    print(f"   - True Positives: {tp}")
    print(f"   - False Positives: {fp}")
    print(f"   - False Negatives: {fn}")
    print(f"   - True Negatives: {tn}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - Accuracy: {accuracy:.4f}")
    
    # Save results
    results_dir = project_root / "results" / "results_anomaly_detection"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        'time_step': range(len(time_step_predictions)),
        'true_label': comparison_labels,
        'predicted_label': time_step_predictions,
        'reconstruction_error': time_step_errors[:len(comparison_labels)]
    })
    results_df.to_csv(results_dir / "detection_results.csv", index=False)
    print(f"\n Saved results to: {results_dir / 'detection_results.csv'}")
    
    # Save metrics
    metrics = {
        'threshold': float(threshold),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy)
    }
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Saved metrics to: {results_dir / 'metrics.json'}")
    
    # ============================================================================
    # PLOTTING
    # ============================================================================
    print("\n Generating plots...")
    
    # Plot 2: Reconstruction Error vs Time (MOST IMPORTANT)
    fig, ax = plt.subplots(figsize=(14, 6))
    time_steps = np.arange(len(time_step_errors))
    ax.plot(time_steps, time_step_errors, 'b-', linewidth=1, alpha=0.7, label='Reconstruction Error')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    
    # Highlight detected anomalies
    anomaly_mask = time_step_predictions == 1
    if np.any(anomaly_mask):
        ax.scatter(time_steps[anomaly_mask], time_step_errors[anomaly_mask], 
                  color='red', s=20, alpha=0.6, zorder=5, label='Detected Anomalies')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Reconstruction Error', fontsize=12)
    ax.set_title('Reconstruction Error vs Time (Anomaly Detection)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = results_dir / "reconstruction_error_vs_time.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {plot_path}")
    
    # Plot 3: Detected vs True Anomalies (Overlay Plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top subplot: True anomalies
    ax1.fill_between(time_steps, 0, comparison_labels, alpha=0.5, color='orange', label='True Anomalies')
    ax1.set_ylabel('True Labels', fontsize=12)
    ax1.set_title('Ground Truth vs Detected Anomalies', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Detected anomalies
    ax2.fill_between(time_steps, 0, time_step_predictions, alpha=0.5, color='red', label='Detected Anomalies')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Predicted Labels', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = results_dir / "detected_vs_true_anomalies.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {plot_path}")
    
    # Plot 5 (Optional): Reconstruction Error Distribution
    # Load training errors if available
    train_errors_path = model_dir / "training_reconstruction_errors.npy"
    if train_errors_path.exists():
        train_errors = np.load(train_errors_path)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(train_errors, bins=50, alpha=0.6, label='Training Errors', color='blue', density=True)
        ax.hist(errors, bins=50, alpha=0.6, label='Test Errors', color='red', density=True)
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
        ax.set_xlabel('Reconstruction Error', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Reconstruction Error Distribution (Training vs Test)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = results_dir / "error_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {plot_path}")
    
    # Plot 4 (Optional): Training Loss Curve
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['training_loss_per_epoch']) + 1)
        ax.plot(epochs, history['training_loss_per_epoch'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = results_dir / "training_loss_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {plot_path}")
    
    print(f"\n All plots saved to: {results_dir}")
