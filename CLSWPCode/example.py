# %%
import numpy as np
from CLSWPCode.CLSWP import CLSWP, CLSWP_missing_data
from wavelets import Haar
# %%
x = np.random.randn(3000)
y = np.random.choice(np.arange(1, 3000), 500, replace=False)
x[y] = np.nan
s = np.linspace(5, 200, 196)

# %%
c = CLSWP_missing_data(x, Haar, s)
I = c.coeffs
h = Haar(s)

# %%
c.compute_ews(0.01, n_iter=100)
# %%
c.plot("Evolutionary Wavelet Spectrum")
# %%
c.plot("Inner Product Kernel")

# %%
c.compute_local_acf(np.arange(0, 100))

# %%
c.plot("Local Autocovariance")

# %%
c.plot("Local Autocorrelation")
# %%
