import numpy as np

from cwt import cwt
from ews import ews
from local_acf import local_autocovariance, local_autocorrelation
from plotting import view

class CLSWP():
    """
    Continuous-time Locally Stationary Wavelet Process (CLSWP)
    
    This class represents a Continuous-time Locally Stationary Wavelet Process (CLSWP).
    It computes the continuous wavelet transform and stores the coefficients.
    It also provides functions to compute the evolutionary wavelet spectrum, local autocovariance,
    and autocorrelation, and view the inner product kernel, ews, local autocovariance, and autocorrelation.
    
    Attributes:
        x (np.ndarray): Data.
        Wavelet (Wavelet Object): Non-initialised wavelet class.
        scales (np.ndarray): Scales, assumed to be regularly spaced.
        sampling_rate (int): Sampling rate.
        bc (str): Boundary conditions - symmetric, periodic, zero, and constant.
        irregularly_spaced (bool): Whether the data is irregularly spaced.
        times (np.ndarray): Time points corresponding to the data  (for irregularly spaced data only).
        min_spacing (float): Minimum spacing between time points (for irregularly spaced data only).
        coeffs (np.ndarray): Coefficients of the continuous wavelet transform.
        A (np.ndarray): Inner product kernel.
        S (list): List to store the evolutionary wavelet spectrum.
        params (list): List to store the parameters used in computing the evolutionary wavelet spectrum.
        local_acf (list): List to store the local autocovariance.
        local_autocorr (list): List to store the local autocorrelation.
        tau (list): List to store the time lags used in computing the local autocovariance and autocorrelation.
    
    Methods:
        compute_ews(mu, method, n_iter, smooth, smooth_wav, by_level):
            Compute the evolutionary wavelet spectrum.
        compute_local_acf(tau, index):
            Compute the local autocovariance and autocorrelation.
        view_A():
            View the inner product kernel.
        view_ews(norm, sqrt, log, index):
            View the evolutionary wavelet spectrum.
        view_local_acf(index):
            View the local autocovariance.
        view_local_autocorr(index):
            View the local autocorrelation.
    """
    
    def __init__(self, x, Wavelet, scales, sampling_rate=1, bc="symmetric",
                 irregularly_spaced=False, times=None, min_spacing=None):
        """
        Initialize the CLSWP object.
        
        Args:
            x (np.ndarray): Data.
            Wavelet (Wavelet Object): Wavelet object.
            scales (np.ndarray): Scales, assumed to be regularly spaced.
            sampling_rate (int): Sampling rate.
            bc (str): Boundary conditions - symmetric, periodic, zero, and constant.
            irregularly_spaced (bool): Whether the data is irregularly spaced.
            times (np.ndarray): Time points corresponding to the data (for irregularly spaced data only).
            min_spacing (float): Minimum spacing between time points (for irregularly spaced data only).
        """
        
        self.x = x
        self.scales = scales
        self.Wavelet = Wavelet(scales)
        self.coeffs = cwt(x, self.Wavelet, sampling_rate=sampling_rate, bc=bc,
                          irregularly_spaced=irregularly_spaced, times=times,
                          min_spacing=min_spacing)
        self.A = self.Wavelet.inner_product_kernel()
        self.S = []
        self.params = []
        self.local_autocov = []
        self.local_autocorr = []
        self.tau = []
    
        
    def compute_ews(self, mu, method="Daubechies_Iter_Asymmetric", n_iter=100, smooth=True,
                    smooth_wav="db4", by_level=True):
        """
        Compute the evolutionary wavelet spectrum.
        
        Args:
            mu (float): Regularisation parameter.
            method (str): Regularisation method to use.
            n_iter (int): Number of iterations (only necessary when using an iterative regularisation method).
            smooth (bool): Whether to smooth the raw wavelet periodogram.
            smooth_wav (str): Smoothing wavelet.
            by_level (bool): Whether to smooth by level or universally.
        """
        S = ews(self.coeffs, self.A, self.scales, mu=mu, method=method, n_iter=n_iter,
                smooth=smooth, wavelet=smooth_wav, by_level=by_level)
        self.S.append(S)
        self.params.append((mu, method, n_iter, smooth, smooth * smooth_wav))

    def compute_local_acf(self, tau, index=-1):
        """
        Compute the local autocovariance and autocorrelation.
        
        Args:
            tau (np.ndarray): Time lags.
            index (int): Index of spectrum to use in self.S.
        """
        S = self.S[index]
        acf = local_autocovariance(S, self.Wavelet, tau)
        self.local_autocov.append(acf)
        self.local_autocorr.append(local_autocorrelation(acf))
        self.tau.append((index, tau))
        
        
    def view_A(self):
        """ View the inner product kernel. """
        view(self.A, self.scales, self.scales, title=self.Wavelet.name + " Inner Product Kernel",
             x_label="Scale", y_label="Scale")
    
    
    def view_ews(self, index=-1):
        """
        View the evolutionary wavelet spectrum.
        
        Args:
            sqrt (bool): Whether to take the square root transform of the spectrum.
            log (bool): Whether to take the log transform of the spectrum.
            index (int): Index of the spectrum to view.
        """
        view(np.log(self.S[index] + 0.0001), np.arange(0, len(self.x)), self.scales,
             title="Evolutionary Wavelet Spectrum", x_label="Time", y_label="Scale")    
        
    def view_local_autocov(self, index=-1):
        """
        View the local autocovariance.
        
        Args:
            index (int): Index of the local autocovariance to view.
        """
        view(self.local_autocov[index], np.arange(0, len(self.x)), self.tau[index][1],
             title="Local Autocovariance", x_label="Time", y_label="Lag")
        
    def view_local_autocorr(self, index=-1):
        """
        View the local autocorrelation.
        
        Args:
            index (int): Index of the local autocorrelation to view.
        """
        view(self.local_autocov[index], np.arange(0, len(self.x)), self.tau[index][1],
             title="Local Autocorrelation", x_label="Time", y_label="Lag")
        
        
        
        
        
        
        
        