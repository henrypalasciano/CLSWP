import numpy as np
from wavelets import Wavelet

def local_autocovariance(S: np.ndarray, Wavelet: Wavelet, tau: np.ndarray) -> np.ndarray:
    """
    Compute the local autocovariance of a time series given its spectrum.
    
    Args:
        S (np.ndarray): Evolutionary wavelet spectrum
        Wavelet (Wavelet): Wavelet to use. This is an instance of the Wavelet class, with the desired scales
        tau (np.ndarray): Time lags of interest
        
    Returns:
        np.ndarray: Local autocovariance
    """
    # Get scales and autocorrelation wavelet
    scales = Wavelet.scales
    Psi = Wavelet.autocorrelation_wavelet(tau).T
    # Compute the local autocovariance
    return (scales[1] - scales[0]) * Psi @ S


def local_autocorrelation(acf: np.ndarray) -> np.ndarray:
    """
    Compute local autocorrelation from local autocovariance.

    Args:
        acf (np.ndarray): Local autocovariance

    Returns:
        np.ndarray: Local autocorrelation
    """
    # Compute the maximum values of the local autocovariance
    max_vals = np.max(acf, axis=0)
    # Avoid division by zero
    max_vals[max_vals == 0] = 1
    # Compute the local autocorrelation
    return  acf / max_vals
