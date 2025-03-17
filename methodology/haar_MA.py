import numpy as np

def haar_MA(T, alpha, sampling_rate):
    """
    Generate a stationary Haar moving average process.

    Args:
        T (float): Length of the time series.
        alpha (float): Length of the moving average window.
        sampling_rate (int): Sampling rate.
    
    Returns:
        ndarray: Haar moving average process.
    """
    # Sample n points from a normal distribution -  driving noise of the process
    n = int((T + alpha) * sampling_rate)
    r_norm = np.random.normal(0, np.sqrt(1 / sampling_rate), n)
    # Construct a Brownian motion
    B = np.cumsum(r_norm)
    # Compute the Haar moving average process
    s_n = int(alpha * sampling_rate)
    x = (B[s_n:] - 2 * B[s_n // 2 : -s_n // 2] + B[:-s_n])
    return x / np.sqrt(alpha)