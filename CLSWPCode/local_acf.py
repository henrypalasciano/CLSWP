import numpy as np
from wavelet_functions import Wavelet


def local_autocovariance(S: np.ndarray, Wavelet: Wavelet, tau: np.ndarray) -> np.ndarray:
    
    """
    Compute the local autocovariance given the spectrum.
    
    Inputs:
        S: np.ndarray - evolutionary wavelet spectrum
        Wavelet: Wavelet - wavelet used
        tau: np.ndarray - time lags of interest
        
    Output:
        acf: np.ndarray - local autocovariance
    """
    scales = Wavelet.scales
    delta = scales[1] - scales[0]
    Psi = Wavelet.autocorrelation_wavelet(tau).T
    
    Psi = Psi
    acf = delta * Psi @ S
    
    return acf


def local_autocorrelation(acf: np.ndarray) -> np.ndarray:
    
    """ 
    Compute local autocorrelation from local autocovariance.

    Inputs:
        acf: np.ndarray - local_autocovariance
    Output:
        ac: np.ndarray - local_autocorrelation
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
