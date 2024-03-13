import numpy as np
import pywt
from wavelet_functions import Wavelet


# ==============================================================================================================
# Evolutionary Wavelet Spectrum and Local Autocovariance
# ==============================================================================================================


def ews(c: np.ndarray, A: np.ndarray, scales: np.ndarray, mu: float = 0.01,
        method: str = "Daubechies_Iter_Asymmetric",  n_iter: int = 100, 
        smooth: bool = True, wavelet: str = "db4", by_level: bool = True) -> np.ndarray:
    
    """
    Function to compute the Evolutionary Wavelet Spectrum (EWS)
    
    Inputs:
        c: np.ndarray -  wavelet coefficients
        A: np.ndarray - inner product kernel
        scales: np.ndarray - scales, assumed to be regularly spaced
        mu: float or np.ndarray - regularisation parameter
        method: str - regularisation method
        n_iter: int - number of iterations
        smooth: bool - whether to smooth the raw wavelet periodogram
        wavelet: str - wavelet to use for performing the smoothing
        by_level: bool - whether to smooth by level or universally
        
    Output:
        S: np.ndarray - evolutionary wavelet spectrum
    """
    
    # Compute delta, the spacing between scales (necessary, since this is a discretisation
    # of a continuous object)
    delta = scales[1] - scales[0]
    # Compute the raw wavelets periodogram and rescale by delta
    I = c ** 2 / delta
    
    # Smooth the raw wavelet periodogram
    if smooth:
        I = smoothing(I, wavelet=wavelet, by_level=by_level)
    
    # Compute the Evolutionary Wavelet Spectrum using one of the soecified methods
    if method == "Daubechies_Iter_Asymmetric":
        S = daub_inv_iter_asym(I, A, mu, n_iter)
    
    elif method == "Tikhonov":
        S = tikhonov(I, A, mu)
    
    elif method == "Lasso":
        S = daub_inv(I, A, mu)
    
    elif method == "Daubechies_Iter":
        S = daub_inv_iter(I, A, mu, n_iter)

    return S

# ==============================================================================================================
# Regularisation Methods
# ==============================================================================================================


def daub_inv_iter_asym(y, A, mu, n_iter):
    
    x = np.random.uniform(0, 1, np.shape(y))
    e = np.real(np.linalg.eig(A)[0][0])
    A = A / e
    
    for i in range(n_iter):
        x = thr_asym(x + A @ (y - A @ x), mu)
        
    return x / e


def thr_asym(x, mu):
    
    x[x < mu / 2] = 0 
    x[x >= mu / 2] -= mu / 2
    
    return x




def tikhonov(y: np.ndarray, A: np.ndarray, mu: float) -> np.ndarray:
    
    """
    Tikhonov regularisation.
    
    Inputs:
        y: np.ndarray - vector or matrix - y = Ax
        A: np.ndarray - matrix operator
        mu: float - regularisation parameter
        
    Output:
        x: np.ndarray - solution of y = Ax
    """
    
    m,n = np.shape(A)
    # Compute inverse of (mu I + A^2)
    B = np.linalg.inv(mu * np.eye(m) + A @ A)
    # Return (mu I + A^2)^-1 A y
    return B @ A @ y



def thr(x, mu):
    
    x[np.abs(x) < mu/2] = 0 
    x[x >= mu/2] -= mu/2
    x[x <= -mu/2] += mu/2
    
    return x
    


def daub_inv(y, A, mu):
    
    eig, ev = np.linalg.eig(A)
    eig = eig.reshape(-1,1)
    coeffs = thr(ev.T @ y * eig, mu) / (eig ** 2)
    coeffs = coeffs.T
    ev = np.array([ev.T])
    x = coeffs @ ev
    
    return x[0].T


def daub_inv_iter_eig(y, A, mu, n_iter):
    
    eig, ev = np.linalg.eig(A)
    x = np.random.uniform(0, 1, np.shape(y))
    evT = np.array([ev.T])
    for i in range(n_iter):
        coeffs = thr(ev.T @ (x + A @ (y - (A @ x))), mu)
        coeffs = coeffs.T
        x = coeffs @ evT
        x = x[0].T
    
    return x



def daub_inv_iter(y, A, mu, n_iter):
    
    x = np.random.uniform(0, 1, np.shape(y))
    e = np.linalg.eig(A)[0][0]
    A = A / e
    
    for i in range(n_iter):
        x = thr(x + A @ (y - A @ x), mu)
        
    return x / e


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