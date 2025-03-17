import numpy as np
from pywt import wavedec, waverec

def smooth(I: np.ndarray, wavelet: str = "db5", thr_estimator: callable = np.std, soft: bool = True, levels: int = 3, by_level: bool = False, log_transform: bool = True) -> np.ndarray:
    """
    Apply wavelet-based smoothing to each level of the raw wavelet periodogram of a locally stationary wavelet process.

    Args:
        I (ndarray): Raw wavelet periodogram.
        wavelet (str, optional): Wavelet used for smoothing. For a list of wavelet families use pywt.families(short=True) and for a list of wavelet names use pywt.wavelist(family=None, kind='all'). For more information see https://pywavelets.readthedocs.io/en/latest/ref/index.html. Defaults to "db5".
        thr_estimator (function, optional): Method used to calculate the threshold. Defaults to np.std.
        soft (bool, optional): If True, apply soft thresholding, else apply hard thresholding. Defaults to True.
        levels (int, optional): Levels to smooth from, with 0 being the coarsest scale. Defaults to 3.
        by_level (bool, optional): If True, apply a different threshold to each scale of the dwt of each level of the raw wavelet periodogram. Defaults to False.
        log_transform (bool, optional): If True, apply a log transform to the raw wavelet periodogram before smoothing. Defaults to True.

    Returns:
        ndarray: Smoothed raw wavelet periodogram.
    """
    # Number of scales and locations
    (m, n) = np.shape(I)
    # Apply a log transform if necessary
    if log_transform:
        I = np.log(I + 1)
    # Compute the discrete wavelet transfrom of each level of the raw wavelet periodogram
    dwt = wavedec(I, wavelet, axis=1)
    # First entry of the dwt is the scaling function coefficient
    levels += 1
    
    if by_level:
        # Apply a different threshold to each scale of the dwt of each level of the raw wavelet periodogram
        for i,c in enumerate(dwt[levels:]):
            t = thr_estimator(c, axis=1, keepdims=True) * np.log(n)
            dwt[i + levels] = thr(c, t, soft)
    else:
        # Compute a single threshold for each level of the raw wavelet periodogram
        t = thr_estimator(np.hstack(dwt[levels:]), axis=1, keepdims=True) * np.log(n)
        for i,c in enumerate(dwt[levels:]):
            dwt[i + levels] = thr(c, t, soft)

    # Invert the initial log transform if necessary
    if log_transform:
        return np.exp(waverec(dwt, wavelet)) - 1
    return waverec(dwt, wavelet)


def thr(x: np.ndarray, t: np.ndarray, soft: bool = True) -> np.ndarray:
    """
    Apply thresholding to an array.

    Args:
        x (ndarray): Input array.
        t (ndarray): Threshold.
        soft (bool, optional): If True, use soft thresholding. If False, use hard thresholding. Defaults to True.

    Returns:
        ndarray: Thresholded array.
    """
    if soft:
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)
    return x * (np.abs(x) > t)


def mad(x: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """
    Calculate the median absolute deviation of an array.

    Args:
        x (ndarray): Input array.
        axis (int, optional): Axis along which to calculate the median absolute deviation. Defaults to 1.

    Returns:
        ndarray: Median absolute deviation of the input array.
    """
    return np.median(np.abs(x - np.median(x, axis=axis, keepdims=keepdims)), axis=axis, keepdims=keepdims) / 0.6745

