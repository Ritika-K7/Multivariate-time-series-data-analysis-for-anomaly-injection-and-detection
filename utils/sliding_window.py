import numpy as np

def create_windows(data, window_size):
    return np.array([
        data[i:i+window_size]
        for i in range(len(data) - window_size)
    ])
