import numpy as np
from wavelet_functions import Wavelet


def local_autocovariance(S: np.ndarray, Wavelet: Wavelet, tau: np.ndarray) -> np.ndarray:
    
    """
    Compute the local autocovariance given the spectrum.
    
    Args:
        S (np.ndarray): Evolutionary wavelet spectrum
        Wavelet (Wavelet): Wavelet used
        tau (np.ndarray): Time lags of interest
        
    Returns:
        np.ndarray: Local autocovariance
    """
    scales = Wavelet.scales
    Psi = Wavelet.autocorrelation_wavelet(tau).T
    
    acf = (scales[1] - scales[0]) * Psi @ S
    
    return acf


def local_autocorrelation(acf: np.ndarray) -> np.ndarray:
    
    """
    Compute local autocorrelation from local autocovariance.

    Args:
        acf (np.ndarray): Local autocovariance

    Returns:
        np.ndarray: Local autocorrelation
    """
    
    ac =  acf / np.max(acf, axis=0)
    
    nan_idx = np.where(np.isnan(ac[0]))[0]
    if len(nan_idx) > 0:
        if nan_idx[0] == 0:
            for idx in nan_idx[::-1]:
                ac[:,idx] = ac[:,idx+1]
        else:
            for idx in nan_idx:
                ac[:,idx] = ac[:,idx-1]
    
    #ac[:, np.isnan(ac[0])] = np.min(ac[:,~np.isnan(ac[0])], axis=1).reshape(-1,1)
    
    return ac
