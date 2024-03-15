import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Wavelet(ABC):
    """
    Abstract base class for wavelet objects.
    """
    @abstractmethod
    def __init__(self, scales):
        self.name = None
        self.scales = scales
    
    @abstractmethod
    def wavelet(self, v):
        pass
    
    @abstractmethod
    def autocorrelation_wavelet(self, tau):
        pass
    
    @abstractmethod
    def inner_product_kernel(self):
        pass
    

class Haar(Wavelet):
    
    def __init__(self, scales):
        
        self.name = "Haar"
        self.scales = scales
    
    def wavelet(self, v, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
            y = np.zeros([len(u), len(v)])
        else:
            y = np.zeros(len(v))
        y[np.logical_and(v > 0, v <= u/2)] = 1
        y[np.logical_and(v > u/2, v <= u)] = -1
        
        return y / np.sqrt(u)
    
    def wavelet_filter(self, sampling_rate):
        
        n_points = np.round(self.scales * sampling_rate).astype(int)
        
        return np.zeros_like(self.scales), self.scales, n_points
    
    
    def autocorrelation_wavelet(self, tau, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
            Psi = np.zeros([len(u), len(tau)])
        else:
            Psi = np.zeros(len(tau))
        frac = np.abs(tau) / u
        Psi[np.abs(tau) <= u/2] = 1 - 3 * frac[np.abs(tau) <= u/2]
        cond = np.logical_and(np.abs(tau) > u/2, np.abs(tau) <= u)
        Psi[cond] = frac[cond] - 1
        
        return Psi 
    
    
    def inner_product_kernel(self):
        
        u = self.scales
        x = u.reshape(-1,1)
        
        A = x ** 2 / (2 * u)
        A[x > u/2] = 0
        l = np.logical_and(x > u/2, x <= u)
        A[l] += (2 * x - u + u ** 2 / (6 * x) - 5 * x ** 2 / (6 * u))[l]
        
        return A + A.T - np.diag(A.diagonal())



class Ricker(Wavelet):
    
    def __init__(self, scales):
        
        self.name = "Ricker"
        self.scales = scales

    def wavelet(self, v, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
        const = 2 / (np.pi ** (1/4) * np.sqrt(3 * u))
        x = v / u
        
        return const * (1 - x ** 2) * np.exp(-(x ** 2) / 2)
    
    def wavelet_filter(self, sampling_rate):
        
        n_points = np.round(4 * self.scales * sampling_rate).astype(int)
        limits = n_points / sampling_rate
        return -limits, limits, 2 * n_points
    
    
    def autocorrelation_wavelet(self, tau, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
        frac = (tau ** 2) / (u ** 2)
        
        return (1 + (frac ** 2) / 12 - frac) * np.exp(-frac / 4)
    
    
    def inner_product_kernel(self):
        
        u = self.scales
        x = u.reshape(-1,1)
        k = (u ** 2 + x ** 2) ** (9/2)
        
        return 70 * np.sqrt(np.pi) * (u * x) ** 5 / (3 * k)


class Shannon(Wavelet):
    
    def __init__(self, scales):
        
        self.name = "Shannon"
        self.scales = scales
    
    def wavelet(self, v, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
        x = np.pi * v / u
        x[x == 0] += 1e-20
        s = (np.sin(2 * x) - np.sin(x)) / (np.sqrt(u) * x)
        
        return s
    
    def wavelet_filter(self, sampling_rate):
        
        n_points = np.round(10 * self.scales * sampling_rate).astype(int)
        limits = n_points / sampling_rate
        return -limits, limits, 2 * n_points
    
    
    def autocorrelation_wavelet(self, tau, u=None):
        
        if u == None:
            u = self.scales.reshape(-1,1)
            return np.sqrt(u) * self.wavelet(tau)
        else:
            return np.sqrt(u) * self.wavelet(tau, u)
    
    
    def inner_product_kernel(self):
        
        u = self.scales
        x = u.reshape(-1,1)
        
        A = 2 * x - u
        A[np.logical_or(x <= u/2, x > u)] = 0
                    
        return A + A.T - np.diag(A.diagonal())











