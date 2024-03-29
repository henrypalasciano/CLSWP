import numpy as np

# ===================================
# Evolutionary Wavelet Spectrum
# ===================================

def ews(I: np.ndarray, A: np.ndarray, scales: np.ndarray, mu: float = 0.01,
        method: str = "Daubechies_Iter_Asymmetric",  n_iter: int = 100) -> np.ndarray:
    """
    Compute the Evolutionary Wavelet Spectrum (EWS).

    Args:
        c (np.ndarray): The raw wavelet periodogram.
        A (np.ndarray): Inner product kernel.
        scales (np.ndarray): Regularly spaced scales.
        mu (float or np.ndarray, optional): Regularisation parameter. Defaults to 0.01.
        method (str, optional): Regularisation method. Defaults to "Daubechies_Iter_Asymmetric".
        n_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Evolutionary wavelet spectrum.
    """
    
    # Compute the Evolutionary Wavelet Spectrum using one of the specified methods
    if method == "Daubechies_Iter_Asymmetric":
        S = daub_inv_iter_asym(I, A, mu, n_iter)
    elif method == "Tikhonov":
        S = tikhonov(I, A, mu)
    elif method == "Lasso":
        S = lasso(I, A, mu)

    return S

# ===================================
# Regularisation Methods
# ===================================

def daub_inv_iter_asym(y: np.ndarray, A: np.ndarray, mu: float, n_iter: int) -> np.ndarray:
    """
    Daubechies Iterative Scheme with Asymmetric Regularisation as adapted from https://arxiv.org/abs/math/0307152.

    Args:
        y (ndarray): Raw wavelet periodogram.
        A (ndarray): Inner product matrix.
        mu (float): Threshold.
        n_iter (int): Number of iterations.

    Returns:
        ndarray: Evolutionary wavelet spectrum.
    """
    # Initialize the solution vector x with random values between 0 and 1
    x = np.random.uniform(0, 1, np.shape(y))
    # Normalize the inner product matrix A by dividing it by the largest eigenvalue
    e = np.real(np.linalg.eig(A)[0][0])
    A = A / e
    # Compute A @ y - mu and A @ A
    A_y = A @ y - mu
    A_2 = A @ A
    
    # Perform the iterative scheme for n_iter iterations
    for i in range(n_iter):
        # Update the solution vector x using the iterative scheme
        x = np.maximum(x + A_y - A_2 @ x, 0)
        
    # Return the normalized solution vector x
    return x / e

def tikhonov(y: np.ndarray, A: np.ndarray, mu: float) -> np.ndarray:
    """
    Tikhonov regularisation as defined in https://arxiv.org/abs/math/0307152.
    
    Args:
        y (np.ndarray): The vector or matrix y = Ax.
        A (np.ndarray): The matrix operator.
        mu (float): The regularization parameter.
        
    Returns:
        np.ndarray: The solution x of y = Ax.
    """
    m, n = np.shape(A)
    # Compute inverse of (mu I + A^2)
    B = np.linalg.inv(mu * np.eye(m) + A @ A)
    # Return (mu I + A^2)^-1 A y
    return B @ A @ y

def lasso(y: np.ndarray, A: np.ndarray, mu: float) -> np.ndarray:
    """
    Lasso regularisation as defined in https://arxiv.org/abs/math/0307152.

    Args:
        y (np.ndarray): The vector or matrix y = Ax.
        A (np.ndarray): The matrix operator.
        mu (float): The regularization parameter.

    Returns:
        np.ndarray: The solution x of y = Ax.
    """  
    # Compute the eigenvalues and eigenvectors of A
    eig, ev = np.linalg.eig(A)
    eig = eig.reshape(-1,1)
    # Compute the coefficients of the solution
    coeffs = threshold(ev.T @ y * eig, mu) / (eig ** 2)
    coeffs = coeffs.T
    ev = np.array([ev.T])
    # Compute the solution
    x = coeffs @ ev
    return x[0].T

def threshold(x: np.ndarray, mu: float) -> np.ndarray:
    """
    Soft thresholding function.

    Args:
        x (np.ndarray): Input array.
        mu (float): Threshold.
    
    Returns:
        np.ndarray: Soft thresholded array.
    """
    # Apply soft thresholding to the input array
    x[np.abs(x) < mu / 2] = 0 
    x[x >= mu / 2] -= mu / 2
    x[x <= - mu / 2] += mu / 2
    return x