import numpy as np

from cwt import cwt
from ews import ews
from local_acf import local_autocovariance, local_autocorrelation
from plotting import view
from wavelets import Wavelet

class CLSWP():
    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray, 
                 sampling_rate: float = 1, bc: str = "symmetric") -> None:
        
        self.ts = ts
        self.scales = scales
        self.Wavelet = Wavelet(scales)
        self.coeffs = cwt(ts, self.Wavelet, sampling_rate=sampling_rate, bc=bc)
        self.A = self.Wavelet.inner_product_kernel()
        self.estimates = []
        
    def compute_ews(self, mu: float, method: str = "Daubechies_Iter_Asymmetric", n_iter: int = 100, 
                    smooth: bool = True, smooth_wav: str = "db4", by_level: bool = True) -> None:
        S = ews(self.coeffs, self.A, self.scales, mu=mu, method=method, n_iter=n_iter,
                smooth=smooth, wavelet=smooth_wav, by_level=by_level)
        self.estimates.append({"ews": S, "mu": mu, "method": method, "n_iter": n_iter})

    def compute_local_acf(self, tau: np.ndarray, index: int = -1):
        acf = local_autocovariance(self.estimates[index]["ews"], self.Wavelet, tau)
        self.estimates[index]["local_autocov"] = acf
        self.estimates[index]["local_autocorr"] = local_autocorrelation(acf)
        self.estimates[index]["tau"] = tau
        
    def plot(self, qty: str, index: int = -1) -> None:
        if qty == "Inner Product Kernel":
            view(self.A, self.scales, self.scales, title=self.Wavelet.name + " Inner Product Kernel",
                 x_label="Scale", y_label="Scale")
        elif qty == "Evolutionary Wavelet Spectrum":
            view(np.log(self.estimates[index]["ews"] + 0.0001), np.arange(0, len(self.ts)), self.scales,
                 title="Evolutionary Wavelet Spectrum", x_label="Time", y_label="Scale")
        elif qty == "Local Autocovariance":
            view(self.estimates[index]["local_autocov"], np.arange(0, len(self.ts)), self.estimates[index]["tau"],
                 title="Local Autocovariance", x_label="Time", y_label="Lag")
        elif qty == "Local Autocorrelation":
            view(self.estimates[index]["local_autocorr"], np.arange(0, len(self.ts)), self.estimates[index]["tau"],
                 title="Local Autocorrelation", x_label="Time", y_label="Lag")
        else:
            raise ValueError("Invalid argument for qty.")
        
class CLSWPMissingData(CLSWP):
    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray, 
                 sampling_rate: float = 1, bc: str = "symmetric") -> None:
        ts = self.reweight(ts)
        super().__init__(ts, Wavelet, scales, sampling_rate, bc)

    @staticmethod
    def reweight(x):
        """
        Reweights the values in the input array based on the number of NaNs nearby.
        This function is used for computing integrals (specifically for the CWT) when values are missing. It assigns weights to each value in the array based on the number of NaNs nearby. NaNs are then replaced by zeros, allowing for a more efficient CWT computation.
        
        Args:
            x (ndarray): Input array containing NaNs.
        
        Returns:
            np.ndarray: The reweighted array, with NaNs replaced by zeros.
        """
        # Find NaNs
        idx = ~np.isnan(x)
        x[~idx] = 0
        # Remove leading and trailing NaNs
        x = np.trim_zeros(x)
        idx = np.trim_zeros(idx)
        # Initialise weights
        weights = idx.copy().astype(float)
        t = np.arange(0, len(x))[idx]
        # Adjust weights based on the spacing between valid observations
        weights[idx] += np.convolve((t[1:] - t[:-1] - 1) / 2, np.array([1, 1]))
        # Reweight the data accordingly
        return x * weights
    
class CLSWPIrregularlySpacedData(CLSWP):
    def __init__(self, ts: np.ndarray, Wavelet: Wavelet, scales: np.ndarray,
                 times: np.ndarray, min_spacing: float, sampling_rate: float = 1, 
                 bc: str = "symmetric") -> None:
        ts = self.spacing_function(ts, times, min_spacing)
        super().__init__(ts, Wavelet, scales, sampling_rate, bc)

    @staticmethod
    def spacing_function(x, times, min_spacing):
        """
        Converts irregularly spaced data to regularly spaced data by adding zeros in between observations and reweighting the time series based on the distance to neighbouring points.

        Args:
            x (np.ndarray): The input time series.
            times (np.ndarray): The corresponding times.
            min_spacing (float): The minimum spacing between any two observations.

        Returns:
            np.ndarray: The regularly spaced data, reweighted based on the distance to neighbouring points and with zeros at the missing locations.
        """
        # Rescale times, so that the first observation is at time 0 and the minimum spacing between any two observations is one
        times = ((times - np.min(times)) / min_spacing).astype(int)
        # Create a new array with NaNs at the missing locations
        y = np.zeros(times[-1] + 1)
        y[times] = (1 + np.convolve((times[1:] - times[:-1] - 1) / 2, np.array([1, 1]))) * x
        # Reweight the time series
        return y