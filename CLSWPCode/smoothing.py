import numpy as np
import pywt

def smoothing(I, wavelet="db10", by_level=True):
    """
    Apply wavelet-based smoothing.

    Args:
        I (ndarray): Raw wavelet periodogram.
        wavelet (str, optional): Wavelet to be used for smoothing. Defaults to "db10".
        by_level (bool, optional): If True, smooth by level. If False, smooth globally. Defaults to True.

    Returns:
        ndarray: Smoothed raw wavelet periodogram.
    """
    # Calculate the dimensions of the input array
    (m, n) = np.shape(I)

    # Decompose the input array using wavelet transform
    coeffs = pywt.wavedec(I, wavelet, axis=1)
    coeffs_stacked = np.hstack(coeffs[1:])

    # Calculate the threshold for wavelet coefficients
    if by_level:
        # Calculate the threshold by level
        t = np.std(coeffs_stacked, axis=1) * np.log(n)
        t = t.reshape(-1,1)
    else:
        # Calculate the global threshold
        t = np.std(coeffs_stacked) * np.log(n)

    # Apply thresholding to the wavelet coefficients
    for i,c in enumerate(coeffs[1:]):
        c[np.abs(c) <= t] = 0
        coeffs[i+1] = c

    # Reconstruct the smoothed wavelet periodogram
    I = pywt.waverec(coeffs, wavelet, axis=1)

    # Return the smoothed wavelet periodogram
    return I