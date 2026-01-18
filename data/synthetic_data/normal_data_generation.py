import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_normal_data(T=1000, noise=0.05):
    t = np.arange(T)

    s1 = np.sin(0.02 * t) + noise * np.random.randn(T)
    s2 = np.sin(0.04 * t + 1) + noise * np.random.randn(T)
    s3 = np.sin(0.01 * t + 2) + noise * np.random.randn(T)

    data = np.stack([s1, s2, s3], axis=1)
    return data

if __name__ == "__main__":
    # Generate the data
    data = generate_normal_data()
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", "results_normal_data_gen")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(data, columns=['s1', 's2', 's3'])
    output_file = os.path.join(results_dir, "normal_data.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Data saved to: {output_file}")
    print(f"Shape: {data.shape}")
    
    # Plot and save the data
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(data[:, i], label=f'Sensor {i+1}')
    plt.legend()
    plt.title("Multivariate Synthetic Time Series")
    plot_file = os.path.join(results_dir, "normal_data_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_file}")
