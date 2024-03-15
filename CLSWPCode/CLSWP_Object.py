import numpy as np
import matplotlib.pyplot as plt

from cwt_functions import cwt
from ews import ews
from local_acf import local_autocovariance, local_autocorrelation
from plotting import view

class CLSWP():
    
    """
    Continuous-time Locally Stationary Wavelet Process (CLSWP)
    
    Inputs:
        x: np.ndaarray - data
        Wavelet: Wavelet Object
        scales: np.ndarray - scales, assumed to be regularly spaced
        sampling rate: int
        bc: boundary conditions - symmetric, periodic, zero and constant
    
    Computes continuous wavelet transform and stores this in self.coeffs
    
    Functions:
        compute_ews - compute the evolutionary wavelet spectrum and store the
        result in self.S. Store the paramteres used in self.params
        compute_local_acf - compute the local autocovariance and autocorrelation 
        and store these is self.local_acf and self.local_autocorr respectively 
        and the time lags used in self.tau
        view_A, view_ews, view_local_acf, view_local_autocorr - view the inner 
        product kernel, ews, local autocovariance and autocorrelation respectively.
        For the latter three, can specify which of these to use via the index
        argument.
        
    The compute_ews and compute_local_acf allow one to compute the quantities 
    several times for different parameter values and store these in corresponding
    arrays for easy access.
    """
    
    def __init__(self, x, Wavelet, scales, sampling_rate=1, bc="symmetric",
                 irregularly_spaced=False, times=None, min_spacing=None):
        
        self.x = x # Data
        self.scales = scales # Scales
        self.Wavelet = Wavelet(scales)
        # Compute Wavelet Transform and Inner Product Kernel for later use
        self.coeffs = cwt(x, self.Wavelet, sampling_rate=sampling_rate, bc=bc,
                          irregularly_spaced=irregularly_spaced, times=times,
                          min_spacing=min_spacing)
        self.A = self.Wavelet.inner_product_kernel()
        
        # Arrays for storing the various ews and corresponding parameters
        self.S = []
        self.params = []
        
        # Arrays for storing the various local acf and autocorrelations and their
        # corresponding parameters
        self.local_acf = []
        self.local_autocorr = []
        self.tau = []
    
        
    def compute_ews(self, mu, method="Daubechies_Iter_Asymmetric", n_iter=100, smooth=True,
                    smooth_wav="db4", by_level=True):
        
        """
        Function for computing the evolutionary wavelet spectrum
        
        Inputs:
            mu: float - regularisation parameter
            method: str - regularisation method to use
            n_iter (optional, default 100): int - number of iterations (only necessary when using a
                                                iterative regularisation method)
            smooth (optional, default True): bool - whether to smooth the raw wavelet periodogram
            smooth_wav (optional, default db4): str - smoothing wavelet
            by_level (optional, default True): bool - whether to smooth by level or universally
        """
        
        S = ews(self.coeffs, self.A, self.scales, mu=mu, method=method, n_iter=n_iter,
                smooth=smooth, wavelet=smooth_wav, by_level=by_level)
        
        self.S.append(S)
        if smooth:
            self.params.append((mu, method, n_iter, smooth, smooth_wav))
        else:
            self.params.append((mu, method, n_iter, smooth, None))

    
    def compute_local_acf(self, tau, index=None):
        
        """
        Function for computing the local acf.
        
        Inputs:
            tau: np.ndarray - time lags
            index (optional, default None): int - index of spectrum to use in self.S
        """
        
        # If no index supplied, use latest ews
        if index == None:
            index = len(self.S) - 1
        S = self.S[index]
        
        # Compute local acf and autocorrelation
        acf = local_autocovariance(S, self.Wavelet, tau)
        local_autocorr = local_autocorrelation(acf)
        
        self.local_acf.append(acf)
        self.local_autocorr.append(local_autocorr)
        self.tau.append((index, tau))
        
        
    def view_A(self):
        """ View the inner product kernel """
        view(self.A, self.scales, which=0, title = self.Wavelet.name + " Inner Product Kernel")
    
    
    def view_ews(self, norm=False, sqrt=False, log=False, index=-1):
        """ View an ews depending on index supplied. The viewed ews can be normalised, 
        square root transfromed or log transformed to improve visibility """
        if sqrt:
            view(np.sqrt(self.S[index]), self.scales, which=1)
        elif log:
            view(np.log(self.S[index]+0.0001), self.scales, which=1)
        elif norm:
            S = self.S[index]
            S = S / np.sum(S, axis=0)
            view(S, self.scales, which=1)
        else:
            view(self.S[index], self.scales, which=1)
        
        
    def view_local_acf(self, index=-1):
        """ View a local acf depending on index supplied. """
        view(self.local_acf[index], self.tau[index][1], which=2)
        
    def view_local_autocorr(self, index=-1):
        """ View a local autocorrelation depending on index supplied. """
        view(self.local_autocorr[index], self.tau[index][1], which=2)
        
        
        
        
        
        
        
        