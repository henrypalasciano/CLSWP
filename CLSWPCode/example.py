# %%
import numpy as np
from CLSWP import CLSWP, CLSWPMissingData, CLSWPIrregularlySpacedData
from wavelets import Haar
# %%
x = np.random.randn(3000)
y = np.random.choice(np.arange(1, 3000), 500, replace=False)
x[y] = np.nan
s = np.linspace(2, 101, 100)
# %%
x = np.random.randn(5000)
s = np.linspace(2, 100, 99)
c0 = CLSWP(x, Haar, s)
# %%
c = CLSWPMissingData(x, Haar, s, keep_all=False)
# %%
c.compute_ews(np.random.randn(1, 5000), n_iter=1000)
# %%
c.plot("Evolutionary Wavelet Spectrum")
# %%
c2 = CLSWPMissingData(x, Haar, s, keep_all=True)
# %%
c2.compute_ews(0.01, n_iter=1000)
# %%
c2.plot("Evolutionary Wavelet Spectrum")
# %%
c.compute_local_acf(np.arange(0, 100))

# %%
c.plot("Local Autocovariance")

# %%
c.plot("Local Autocorrelation")
# %%
x = np.random.randn(3000)
t = np.linspace(0, 3000, 6000)
times = np.sort(np.round(np.random.choice(t, 3000, replace=False), 1))
s = np.linspace(2, 201, 200)
# %%
c = CLSWPIrregularlySpacedData(x, Haar, s, times, sampling_rate=0.5, keep_all=False)
# %%
c.compute_ews(0.01, n_iter=1000)
# %%
c.plot("Evolutionary Wavelet Spectrum")
# %%

# %%
