import numpy as np
from typing import Union
from pywt import wavedec
from smoothing import mad

# ===================================
# Evolutionary Wavelet Spectrum
# ===================================

def ews(I: np.ndarray, A: np.ndarray, mu: Union[float, np.ndarray] = None, measure: callable = mad, mu_wav: str = "db5",  
        N: int = 100, S: np.array = None, u_idx: int = None) -> np.ndarray:
    """
    Compute the Evolutionary Wavelet Spectrum (EWS).

    Args:
        I (np.ndarray): The raw wavelet periodogram.
        A (np.ndarray): Inner product kernel.
        mu (float or np.ndarray, optional): Regularisation parameter. Defaults to None.
        measure (callable, optional): Measure function. Defaults to mad. Only used if mu is None.
        mu_wav (str, optional): Wavelet used for the measure function. Defaults to "db5". Only used if mu is None.
        N (int, optional): Number of iterations. Defaults to 100.
        S (np.array, optional): Initial solution evolutionary wavelet spectrum. Defaults to None.
        u_idx (int, optional): Index of widest scale to update to. Defaults to None.

    Returns:
        np.ndarray: Evolutionary wavelet spectrum.
    """
    # If none 
    if mu is None:
        # Finest scale wavelet coefficients at each scale
        dwt = wavedec(I, wavelet=mu_wav, axis=1, level=1)[-1]
        mu = measure(dwt, axis=1, keepdims=True)

    if u_idx is None:
        # Normalize the inner product matrix A by dividing it by the largest eigenvalue
        e = np.real(np.linalg.eig(A)[0][0])
    else:
        e = np.real(np.linalg.eig(A[:u_idx, :u_idx])[0][0])
    A = A / e
    I = I / e
    # Initialize the solution vector x with random values between 0 and 1
    if S is None:
        S = I.copy()
    # Compute A @ y - mu and A @ A
    A_y = A @ I - mu
    A_2 = A @ A
    # Run the iterative scheme for N iterations
    if u_idx is None:
        for i in range(N):
            # Update the solution vector x using the iterative scheme
            S = np.maximum(S + A_y - A_2 @ S, 0)
    else:
        for i in range(N):
            # Update the solution vector x using the iterative scheme
            S[:u_idx] = np.maximum(S[:u_idx] + A_y[:u_idx] - A_2[:u_idx] @ S, 0)

    return S