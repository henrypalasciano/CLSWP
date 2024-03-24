import numpy as np
from abc import ABC, abstractmethod
class Wavelet(ABC):
    """
    Methods:
        __init__(self, scales): Initialize the Wavelet object with the given scales.
        wavelet(self, v): Compute the wavelet values at v.
        wavelet_filter(self, sampling_rate): Determine an interval and number of points for sampling the wavelet at each scale.
        autocorrelation_wavelet(self, tau): Compute the autocorrelation of the wavelet function at lags tau.
        inner_product_kernel(self): Compute the inner product kernel for the given scales.
    """
    @abstractmethod
    def __init__(self, scales):
        self.name = None
        self.scales = scales
    
    @abstractmethod
    def wavelet(self, v):
        pass
    
    @abstractmethod
    def wavelet_filter(self, sampling_rate):
        pass
    
    @abstractmethod
    def autocorrelation_wavelet(self, tau):
        pass
    
    @abstractmethod
    def inner_product_kernel(self):
        pass
    
class Haar(Wavelet):
    """
    Haar Wavelet Class.
    """
    def __init__(self, scales: np.ndarray) -> None:
        self.name = "Haar"
        self.scales = scales
    
    def wavelet(self, v: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        # If u is not provided, carry out computation for all scales
        if u == None:
            u = self.scales.reshape(-1,1)
            y = np.zeros([len(u), len(v)])
        else:
            y = np.zeros(len(v))
        # Compute the Haar wavelet at scales u
        y[np.logical_and(v > 0, v <= u/2)] = 1
        y[np.logical_and(v > u/2, v <= u)] = -1
        return y / np.sqrt(u)
    
    def wavelet_filter(self, sampling_rate: float) -> tuple:
        # Compute the limits and number of points for the Haar wavelet
        n_points = np.round(self.scales * sampling_rate).astype(int)
        return np.zeros_like(self.scales), self.scales, n_points
    
    def autocorrelation_wavelet(self, tau: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        # If u is not provided, carry out computation for all scales
        if u == None:
            u = self.scales.reshape(-1,1)
            Psi = np.zeros([len(u), len(tau)])
        else:
            Psi = np.zeros(len(tau))
        # Compute the Haar autocorrelation wavelet at scales u
        frac = np.abs(tau) / u
        Psi[np.abs(tau) <= u/2] = 1 - 3 * frac[np.abs(tau) <= u/2]
        cond = np.logical_and(np.abs(tau) > u/2, np.abs(tau) <= u)
        Psi[cond] = frac[cond] - 1
        return Psi 
    
    def inner_product_kernel(self) -> np.ndarray:
        # Scales at which to compute the inner product kernel
        u = self.scales
        x = u.reshape(-1,1)
        # Compute the inner product kernel for the Haar wavelet
        A = x ** 2 / (2 * u)
        A[x > u/2] = 0
        l = np.logical_and(x > u/2, x <= u)
        A[l] += (2 * x - u + u ** 2 / (6 * x) - 5 * x ** 2 / (6 * u))[l]
        return A + A.T - np.diag(A.diagonal())

class Ricker(Wavelet):
    """
    Ricker Wavelet Class.
    """
    class Ricker(Wavelet):
        """
        Ricker Wavelet Class.
        """
        def __init__(self, scales: np.ndarray) -> None:
            self.name = "Ricker"
            self.scales = scales
        
        def wavelet(self, v: np.ndarray, u: np.ndarray = None) -> np.ndarray:
            # If u is not provided, carry out computation for all scales
            if u == None:
                u = self.scales.reshape(-1,1)
            # Compute the Ricker wavelet at scales u
            const = 2 / (np.pi ** (1/4) * np.sqrt(3 * u))
            x = v / u
            return const * (1 - x ** 2) * np.exp(-(x ** 2) / 2)
        
        def wavelet_filter(self, sampling_rate: float) -> tuple:
            # Compute the limits and number of points for the Ricker wavelet
            n_points = np.round(4 * self.scales * sampling_rate).astype(int)
            limits = n_points / sampling_rate
            return -limits, limits, 2 * n_points
        
        def autocorrelation_wavelet(self, tau: np.ndarray, u: np.ndarray = None) -> np.ndarray:
            # If u is not provided, carry out computation for all scales
            if u == None:
                u = self.scales.reshape(-1,1)
            # Compute the Ricker autocorrelation wavelet at scales u
            frac = (tau ** 2) / (u ** 2)
            return (1 + (frac ** 2) / 12 - frac) * np.exp(-frac / 4)
        
        def inner_product_kernel(self) -> np.ndarray:
            # Scales at which to compute the inner product kernel
            u = self.scales
            x = u.reshape(-1,1)
            # Compute the inner product kernel for the Ricker wavelet
            k = (u ** 2 + x ** 2) ** (9/2)
            return 70 * np.sqrt(np.pi) * (u * x) ** 5 / (3 * k)
class Shannon(Wavelet):
    """
    Shannon Wavelet Class.
    """
    def __init__(self, scales: np.ndarray) -> None:
        self.name = "Shannon"
        self.scales = scales

    def wavelet(self, v: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        # If u is not provided, carry out computation for all scales
        if u == None:
            u = self.scales.reshape(-1,1)
        # Compute the Shannon wavelet at scales u
        x = np.pi * v / u
        x[x == 0] += 1e-20
        s = (np.sin(2 * x) - np.sin(x)) / (np.sqrt(u) * x)
        return s

    def wavelet_filter(self, sampling_rate: float) -> tuple:
        # Compute the limits and number of points for the Shannon wavelet
        n_points = np.round(10 * self.scales * sampling_rate).astype(int)
        limits = n_points / sampling_rate
        return -limits, limits, 2 * n_points

    def autocorrelation_wavelet(self, tau: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        # If u is not provided, carry out computation for all scales
        if u == None:
            u = self.scales.reshape(-1,1)
            # Compute the Shannon autocorrelation wavelet at scales u
            return np.sqrt(u) * self.wavelet(tau)
        else:
            # Compute the Shannon autocorrelation wavelet at scales u
            return np.sqrt(u) * self.wavelet(tau, u)

    def inner_product_kernel(self) -> np.ndarray:
        # Scales at which to compute the inner product kernel
        u = self.scales
        x = u.reshape(-1,1)
        # Compute the inner product kernel for the Shannon wavelet
        A = 2 * x - u
        A[np.logical_or(x <= u/2, x > u)] = 0       
        return A + A.T - np.diag(A.diagonal())
    
    
Haar.__doc__ += Wavelet.__doc__
Ricker.__doc__ += Wavelet.__doc__
Shannon.__doc__ += Wavelet.__doc__