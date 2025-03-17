# %%
import numpy as np
import matplotlib.pyplot as plt
from CLSWP import CLSWP, CLSWPMissingData, CLSWPIrregularlySpacedData
from wavelets import Haar
from haar_MA import haar_MA
from plotting import view
# %%
# Regularly spaced data example
x = haar_MA(10, 1, 100)
scales = np.linspace(0.1, 10, 100)
# Initialise CLSWP object
C1 = CLSWP(x, Haar, scales, sampling_rate=100)
# Compute the Evolutionary Wavelet Spectrum and plot this
C1.smooth_rwp(smooth_wav="db3")
C1.compute_ews(N=1000)
C1.plot("Evolutionary Wavelet Spectrum")
# %%
C1.compute_ews(N=1000, update=True, u_idx=50)
C1.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C1.compute_local_acf(np.linspace(0, 5, 101))
C1.plot("Local Autocovariance")
# %%
# Missing data example
y = np.random.choice(np.arange(0, 1000), 250, replace=False)
z = x.copy()
z[y] = np.nan
# Initialise CLSWPMissingData object with keep all set to False (only computes the EWS at original data locations, speeding up the computation)
C2 = CLSWPMissingData(z, Haar, scales, sampling_rate=100, keep_all=False)
# Compute the Evolutionary Wavelet Spectrum and plot this
C2.compute_ews(N=1000)
C2.plot("Evolutionary Wavelet Spectrum")
# %%
C2.compute_ews(N=1000, update=True, u_idx=50)
C2.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C2.compute_local_acf(np.linspace(0, 5, 101))
C2.plot("Local Autocovariance")
# %%
# Irregularly spaced data example
obs = np.random.binomial(1, 0.75, 1000).astype(bool)
t = np.arange(0, 10, 0.01)[obs]
C3 = CLSWPIrregularlySpacedData(x[obs], Haar, scales, t, sampling_rate=100, keep_all=True)
# Compute the Evolutionary Wavelet Spectrum and plot this
C3.compute_ews(N=1000)
C3.plot("Evolutionary Wavelet Spectrum")
# %%
C3.compute_ews(N=1000, update=True, u_idx=50)
C3.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C3.compute_local_acf(np.arange(0, 100))
C3.plot("Local Autocovariance")
# %%
