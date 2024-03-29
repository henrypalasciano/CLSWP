import numpy as np

from cwt import cwt
from ews import ews
from local_acf import local_autocovariance, local_autocorrelation
from plotting import view
from smoothing import wavelet_smoothing
from wavelets import Wavelet

class CLSWP():
    """
    Continuous-time Locally Stationary Wavelet Process (CLSWP) Object.

    This class represents a CLSWP object that can be used to compute various properties and perform analysis on a given time series.

    Attributes:
        ts (np.ndarray): The time series data.
        Wavelet (Wavelet): The wavelet object.
        scales (np.ndarray): The scales for the continuous wavelet transform.
        sampling_rate (float): The sampling rate of the time series data.
        bc (str): The boundary condition for the continuous wavelet transform.
        coeffs (np.ndarray): The continuous wavelet transform coefficients.
        A (np.ndarray): The inner product kernel.
        regular (bool): Whether the ews and other estimates lie on a regularly spaced grid. Required for the plotting function.
        times (np.ndarray): The times at which the observations are made.
        estimates (list): List of dictionaries containing computed estimates with different parameters.

    Methods:
        compute_ews: Compute the Evolutionary Wavelet Spectrum (EWS) for a given value of mu, method and number of iterations (if applicable).
        compute_local_acf: Compute the local autocovariance and local autocorrelation for a given set of lags, tau.
        plot: Plot various quantities related to the CLSWP object.
    """

    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray, 
                 sampling_rate: float = 1, bc: str = "symmetric") -> None:
        """
        Initialize the CLSWP Object.

        Args:
            ts (np.ndarray): The time series data.
            Wavelet (Wavelet): The wavelet object.
            scales (np.ndarray): The scales for the continuous wavelet transform.
            sampling_rate (float, optional): The sampling rate of the time series data. Defaults to 1.
            bc (str, optional): The boundary condition for the continuous wavelet transform. Defaults to "symmetric".
        """
        self.ts = ts
        self.scales = scales
        self.Wavelet = Wavelet(scales)
        self.coeffs = cwt(ts, self.Wavelet, sampling_rate=sampling_rate, bc=bc)
        # Compute the inner product kernel
        self.A = self.Wavelet.inner_product_kernel()
        # Whether the ews and other estimates lie on a regularly spaced grid. Required for the plotting function in the child classes.
        self.regular = True
        # The times at which the observations are made
        self.times = np.arange(0, len(self.ts))
        self.sampling_rate = sampling_rate
        # List of dictionaries containing computed estimates with different parameters
        self.estimates = []
        
    def compute_ews(self, mu: float, method: str = "Daubechies_Iter_Asymmetric", n_iter: int = 100, 
                    smooth: bool = True, smooth_wav: str = "db4", by_level: bool = True) -> None:
        """
        Compute the Evolutionary Wavelet Spectrum (EWS) for a given value of mu, method and number of iterations (if applicable).

        Args:
            mu (float): The mu value for computing the EWS.
            method (str, optional): The method for computing the EWS. Defaults to "Daubechies_Iter_Asymmetric".
            n_iter (int, optional): The number of iterations for computing the EWS. Defaults to 100.
            smooth (bool, optional): Whether to apply smoothing to the raw wavelet periodogram before estimating the EWS. Defaults to True.
            smooth_wav (str, optional): The wavelet for smoothing the raw wavelet periodogram. Defaults to "db4".
            by_level (bool, optional): Whether to perform the smoothing by level or globally. Defaults to True.
        """
        # Compute the raw wavelet periodogram
        I = self.coeffs ** 2
        # Apply smoothing to the raw wavelet periodogram
        if smooth:
            I = wavelet_smoothing(I, wavelet=smooth_wav, by_level=by_level)
        # If not regular, keep only locations corresponding to observed values
        if not self.regular:
            I = I[:, self.times]
        S = ews(I, self.A, self.scales, mu=mu, method=method, n_iter=n_iter)
        self.estimates.append({"ews": S, "mu": mu, "method": method, "n_iter": n_iter})

    def compute_local_acf(self, tau: np.ndarray, index: int = -1):
        """
        Compute the local autocovariance and local autocorrelation for a given set of lags, tau.

        Args:
            tau (np.ndarray): The lags for which to compute the local autocovariance and local autocorrelation.
            index (int, optional): The index of the EWS estimate to use. Defaults to -1.
        """
        acf = local_autocovariance(self.estimates[index]["ews"], self.Wavelet, tau)
        self.estimates[index]["local_autocov"] = acf
        self.estimates[index]["local_autocorr"] = local_autocorrelation(acf)
        self.estimates[index]["tau"] = tau
        
    def plot(self, qty: str, index: int = -1, num_x: int = 5, num_y: int = 5) -> None:
        """
        Plot various quantities related to the CLSWP object.

        Args:
            qty (str): The quantity to plot. Valid values are "Inner Product Kernel", "Evolutionary Wavelet Spectrum",
                       "Local Autocovariance", and "Local Autocorrelation".
            index (int, optional): The index of the estimate to use. Defaults to -1.
            num_x (int, optional): The number of x-axis ticks. Defaults to 5.
            num_y (int, optional): The number of y-axis ticks. Defaults to 5.
        """
        times = self.times * self.sampling_rate
        if qty == "Inner Product Kernel":
            view(self.A, self.scales, self.scales, regular=True, title=self.Wavelet.name + " " + qty,
                 x_label="Scale", y_label="Scale", num_x=num_x, num_y=num_y)
        elif qty == "Evolutionary Wavelet Spectrum":
            view(np.log(self.estimates[index]["ews"] + 0.0001), times, self.scales, regular=self.regular,
                 title=qty, x_label="Time", y_label="Scale", num_x=num_x, num_y=num_y)
        elif qty == "Local Autocovariance":
            view(self.estimates[index]["local_autocov"], times, self.estimates[index]["tau"],
                 regular=self.regular, title=qty, x_label="Time", y_label="Lag", num_x=num_x, num_y=num_y)
        elif qty == "Local Autocorrelation":
            view(self.estimates[index]["local_autocorr"], times, self.estimates[index]["tau"],
                 regular=self.regular, title=qty, x_label="Time", y_label="Lag", num_x=num_x, num_y=num_y)
        else:
            raise ValueError("Invalid argument for qty.")
        
class CLSWPMissingData(CLSWP):    
    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray, sampling_rate: float = 1, 
                 bc: str = "symmetric", keep_all: bool = True) -> None:
        """
        Initialize the CLSWP_Object for time series containing missing observations.

        Parameters:
        ts (np.ndarray): The time series data.
        Wavelet (Wavelet): The wavelet object.
        scales (np.ndarray): The scales for wavelet decomposition.
        sampling_rate (float, optional): The sampling rate of the time series. Defaults to 1.
        bc (str, optional): The boundary condition for wavelet decomposition. Defaults to "symmetric".
        keep_all (bool, optional): Whether to keep all coefficients or only the ones corresponding to non-missing values. Defaults to True.
        """
        ts, times = self.reweight(ts)
        super().__init__(ts, Wavelet, scales, sampling_rate, bc)
        self.regular = keep_all
        if keep_all:
            self.times = np.arange(0, len(ts))
        else:
            # Irregularly spaced times
            self.times = times
            
    @staticmethod
    def reweight(x):
        """
        Reweights the values in the input array based on the number of NaNs nearby.
        This function is used for computing integrals (specifically for the CWT) when values are missing. It assigns weights to each value in the array based on the number of NaNs nearby. NaNs are then replaced by zeros, allowing for a more efficient CWT computation.
        
        Args:
            x (ndarray): Input array containing NaNs.
        
        Returns:
            tuple: A tuple containing:
                np.ndarray: The reweighted array, with NaNs replaced by zeros.
                np.ndarray: The array of indices of the valid observations in the input array.
        """
        # Find NaNs
        idx = ~np.isnan(x)
        x[~idx] = 0
        # Remove leading and trailing NaNs
        x = np.trim_zeros(x)
        idx = np.trim_zeros(idx)
        # Initialise weights
        weights = idx.copy().astype(float)
        times = np.arange(0, len(x))[idx]
        # Adjust weights based on the spacing between valid observations
        weights[idx] += np.convolve((times[1:] - times[:-1] - 1) / 2, np.array([1, 1]))
        # Reweight the data accordingly
        return x * weights, times
    
class CLSWPIrregularlySpacedData(CLSWP):
    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray, times: np.ndarray, 
                 sampling_rate: float, bc: str = "symmetric", keep_all: bool = True) -> None:
            """
            Initialize the CLSWP_Object for irregularly spaced data.

            Parameters:
            ts (np.ndarray): The time series data.
            Wavelet (Wavelet): The wavelet object.
            scales (np.ndarray): The scales for the wavelet transform.
            times (np.ndarray): The times corresponding to the observations.
            sampling_rate (float): The sampling rate of the time series data. For irregularly spaced data, this is the spacing between observations for the corresponding time series with equally spaced observations.
            bc (str, optional): The boundary condition for the wavelet transform. Defaults to "symmetric".
            keep_all (bool, optional): Whether to keep all coefficients or only the ones corresponding to non-missing values. Defaults to True.
            """
            ts, times = self.spacing_function(ts, times, sampling_rate)
            super().__init__(ts, Wavelet, scales, 1, bc)
            self.regular = keep_all
            if keep_all:
                self.times = np.arange(0, len(ts))
            else:
                # Keep only the coefficients corresponding to non-missing values
                self.times = times

    @staticmethod
    def spacing_function(x, times, sampling_rate):
        """
        Converts irregularly spaced data to regularly spaced data by adding zeros in between observations and reweighting the time series based on the distance to neighbouring points.

        Args:
            x (np.ndarray): The input time series.
            times (np.ndarray): The corresponding times.
            sampling_rate (float): The sampling rate of the time series data. For irregularly spaced data, this is the spacing between observations for the corresponding time series with equally spaced observations.

        Returns:
            np.ndarray: The regularly spaced data, reweighted based on the distance to neighbouring points and with zeros at the missing locations.
        """
        # Rescale times, so that the first observation is at time 0 and the minimum spacing between any two observations is one
        times = ((times - times[0]) / sampling_rate).astype(int)
        # Create a new array with NaNs at the missing locations
        y = np.zeros(times[-1] + 1)
        y[times] = (1 + np.convolve((times[1:] - times[:-1] - 1) / 2, np.array([1, 1]))) * x
        # Reweight the time series
        return y, times