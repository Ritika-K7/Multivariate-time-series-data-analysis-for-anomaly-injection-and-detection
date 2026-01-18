import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# ANOMALY INJECTION FUNCTIONS
# ============================================================================

def inject_spike_anomaly(data):
    labels = np.zeros(len(data))
    data[400:420, 1] += 3
    labels[400:420] = 1
    return data, labels

def inject_drift_anomaly(data):
    labels = np.zeros(len(data))
    # Drift anomaly (collective shift)
    data[700:750, :] -= 2
    labels[700:750] = 1
    return data, labels

def inject_noise_burst_anomaly(data):
    labels = np.zeros(len(data))
    data[550:580, 2] += np.random.normal(0, 1.5, size=30)
    labels[550:580] = 1
    return data, labels

def inject_flatline_anomaly(data):
    labels = np.zeros(len(data))
    data[820:860, 1] = data[820, 1]
    labels[820:860] = 1
    return data, labels

def inject_all_anomalies(data):
    labels = np.zeros(len(data))
    
    # Spike anomaly
    data[400:420, 1] += 3
    labels[400:420] = 1
    
    # Drift anomaly (collective)
    data[700:750, :] -= 2
    labels[700:750] = 1
    
    # Noise burst
    data[550:580, 2] += np.random.normal(0, 1.5, size=30)
    labels[550:580] = 1
    
    # Flatline anomaly
    data[820:860, 1] = data[820, 1]
    labels[820:860] = 1
    
    return data, labels


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_anomaly_data(data_with_anomalies, labels, title, save_path):
    plt.figure(figsize=(14, 8))
    colors = ['blue', 'orange', 'green']
    for i in range(3):
        plt.plot(data_with_anomalies[:, i], label=f'Sensor {i+1}', color=colors[i], alpha=0.7)
        # Highlight anomalies in red
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, data_with_anomalies[anomaly_indices, i], 
                       color='red', s=10, alpha=0.5, zorder=5)
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load normal data
    normal_data_file = os.path.join("results", "results_normal_data_gen", "normal_data.csv")
    df_normal = pd.read_csv(normal_data_file)
    normal_data = df_normal[['s1', 's2', 's3']].values
    
    print(f"Loaded normal data from: {normal_data_file}")
    print(f"Normal data shape: {normal_data.shape}")
    
    # Create results directory
    results_dir = os.path.join("results", "results_anomaly_data_generation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate individual anomaly plots
    anomaly_types = [
        ("spike_only", inject_spike_anomaly, "Spike Anomaly"),
        ("drift_only", inject_drift_anomaly, "Drift Anomaly"),
        ("noise_burst_only", inject_noise_burst_anomaly, "Noise Burst Anomaly"),
        ("flatline_only", inject_flatline_anomaly, "Flatline Anomaly")
    ]
    
    for filename_prefix, inject_func, title in anomaly_types:
        # Create fresh copy for each anomaly type
        data_copy = normal_data.copy()
        data_anomaly, labels = inject_func(data_copy)
        
        # Save CSV
        df_anomaly = pd.DataFrame(data_anomaly, columns=['s1', 's2', 's3'])
        df_anomaly['label'] = labels
        csv_file = os.path.join(results_dir, f"{filename_prefix}.csv")
        df_anomaly.to_csv(csv_file, index=False)
        
        # Save plot
        plot_file = os.path.join(results_dir, f"{filename_prefix}.png")
        plot_anomaly_data(data_anomaly, labels, title, plot_file)
        
        print(f"Generated {filename_prefix}: {int(labels.sum())} anomalies")
    
    # Generate combined/mixed anomaly plot
    data_mixed = normal_data.copy()
    data_mixed, labels_mixed = inject_all_anomalies(data_mixed)
    
    # Save mixed anomaly CSV
    df_mixed = pd.DataFrame(data_mixed, columns=['s1', 's2', 's3'])
    df_mixed['label'] = labels_mixed
    mixed_csv_file = os.path.join(results_dir, "mixed_anomaly.csv")
    df_mixed.to_csv(mixed_csv_file, index=False)
    
    # Save mixed anomaly plot
    mixed_plot_file = os.path.join(results_dir, "mixed_anomaly.png")
    plot_anomaly_data(data_mixed, labels_mixed, "Mixed Anomalies (All Types)", mixed_plot_file)
    
    print(f"\nMixed anomalies: {int(labels_mixed.sum())} total anomalies")
    print(f"All files saved to: {results_dir}")
