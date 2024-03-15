import numpy as np
import pywt

# ==============================================================================================================
# Smoothing
# ==============================================================================================================


def smoothing(I, wavelet="db10", by_level=True):
    
    (m, n) = np.shape(I)
    coeffs = pywt.wavedec(I, wavelet, axis=1)
    coeffs_stacked = np.hstack(coeffs[1:])
    
    if by_level:
        t = np.std(coeffs_stacked, axis=1) * np.log(n)
        t = t.reshape(-1,1)
            
    else:
        t = np.std(coeffs_stacked) * np.log(n)
        
    for i,c in enumerate(coeffs[1:]):
        c[np.abs(c) <= t] = 0
        coeffs[i+1] = c
    
    I = pywt.waverec(coeffs, wavelet, axis=1)
    
    return I