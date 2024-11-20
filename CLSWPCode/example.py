# %%
import numpy as np
from CLSWP import CLSWP, CLSWPMissingData, CLSWPIrregularlySpacedData
from wavelets import Haar
# %%
# Regularly spaced data example
x = np.random.randn(5000)
scales = np.linspace(1, 100, 100)
# Initialise CLSWP object
C1 = CLSWP(x, Haar, scales)
# Compute the Evolutionary Wavelet Spectrum and plot this
C1.compute_ews(N=1000)
C1.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C1.compute_local_acf(np.arange(0, 100))
C1.plot("Local Autocovariance")
# %%
# Missing data example
y = np.random.choice(np.arange(0, 5000), 750, replace=False)
z = x.copy()
z[y] = np.nan
# Initialise CLSWPMissingData object with keep all set to False (only computes the EWS at original data locations, speeding up the computation)
C2 = CLSWPMissingData(z, Haar, scales, keep_all=False)
# Compute the Evolutionary Wavelet Spectrum and plot this
C2.compute_ews(N=1000)
C2.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C2.compute_local_acf(np.arange(0, 100))
C2.plot("Local Autocovariance")
# %%
# Irregularly spaced data example
t = np.random.choice(np.arange(0, 7000), 5000, replace=False)
t = np.sort(t)
C3 = CLSWPIrregularlySpacedData(x, Haar, scales, t, sampling_rate=1)
# Compute the Evolutionary Wavelet Spectrum and plot this
C3.compute_ews(N=1000)
C3.plot("Evolutionary Wavelet Spectrum")
# %%
# Compute the local autocovariance and autocorrelation and plot the local autocovariance
C3.compute_local_acf(np.arange(0, 100))
C3.plot("Local Autocovariance")
# %%
