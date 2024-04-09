import numpy as np
from pywt import wavedec, waverec

def wav_periodogram_smoothing(I, wavelet="db5", soft=True, levels=3):
    """
    Apply wavelet-based smoothing to each level of the raw wavelet periodogram of a locally stationary wavelet process.

    Args:
        I (ndarray): Raw wavelet periodogram.
        wavelet (str, optional): Wavelet to be used for smoothing. For a list of wavelet families use pywt.families(short=True) and for a list of wavelet names use pywt.wavelist(family=None, kind='all'). For more information see https://pywavelets.readthedocs.io/en/latest/ref/index.html. Defaults to "db5".
        soft (bool, optional): If True, use soft thresholding. If False, use hard thresholding. Defaults to True.
        levels (int, optional): Finest level of wavelet decomposition not smoothed from. For example levels=3 smoothes the coefficients from the 4rd level onwards, where level 1 is the coarsest scale. Defaults to 3.

    Returns:
        ndarray: Smoothed raw wavelet periodogram.
    """
    # Calculate the dimensions of the input array
    (m, n) = np.shape(I)
    # Decompose the input array using discrete wavelet transform
    coeffs = wavedec(I, wavelet, axis=1)
    coeffs_stacked = np.hstack(coeffs[levels+1:])
    # Calculate the threshold for each level of the raw wavelet periodogram
    t = (np.std(coeffs_stacked, axis=1) * np.log(n)).reshape(-1,1)
    # Apply thresholding to the wavelet coefficients
    if soft:
        for i,c in enumerate(coeffs[levels+1:]):
            c = np.sign(c) * np.maximum(np.abs(c) - t, 0)
            coeffs[i+levels+1] = c
    else:
        for i,c in enumerate(coeffs[levels+1:]):
            c = c * (np.abs(c) > t)
            coeffs[i+levels+1] = c
    # Return the smoothed wavelet periodogram
    return waverec(coeffs, wavelet)


def wavelet_smoothing(d, wavelet="db5", soft=True, levels=3):
    """
    Apply wavelet-based smoothing to each level of the wavelet coefficients of a locally stationary wavelet process.

    Args:
        d (ndarray): Wavelet coefficients.
        wavelet (str, optional): Wavelet to be used for smoothing. For a list of wavelet families use pywt.families(short=True) and for a list of wavelet names use pywt.wavelist(family=None, kind='all'). For more information see https://pywavelets.readthedocs.io/en/latest/ref/index.html. Defaults to "db5".
        soft (bool, optional): If True, use soft thresholding. If False, use hard thresholding. Defaults to True.
        levels (int, optional): Finest level of wavelet decomposition not smoothed from. For example levels=3 smoothes the coefficients from the 4rd level onwards, where level 1 is the coarsest scale. Defaults to 3.

    Returns:
        ndarray: Smoothed raw wavelet periodogram.
    """
    # Calculate the dimensions of the input array
    (m, n) = np.shape(d)
    # Decompose the input array using discrete wavelet transform
    coeffs = wavedec(d, wavelet, axis=1)
    coeffs_stacked = np.hstack(coeffs[levels+1:])
    # Calculate the threshold for each level of the raw wavelet periodogram
    sigma = np.median(np.abs(coeffs_stacked - np.median(coeffs_stacked))) / 0.6745
    t = sigma.reshape(-1,1) * np.sqrt(2 * np.log(n))
    # Apply thresholding to the wavelet coefficients
    if soft:
        for i,c in enumerate(coeffs[levels+1:]):
            c = np.sign(c) * np.maximum(np.abs(c) - t, 0)
            coeffs[i+levels+1] = c
    else:
        for i,c in enumerate(coeffs[levels+1:]):
            c = c * (np.abs(c) > t)
            coeffs[i+levels+1] = c
    # Return the smoothed wavelet periodogram
    return waverec(coeffs, wavelet)