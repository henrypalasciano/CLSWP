# %%
import numpy as np
from CLSWP_Object import CLSWP
from wavelets import Haar
from cwt import cwt_arbitrary_shifts

x = np.random.randn(3000)
s = np.linspace(5, 200, 196)

# %%
c = CLSWP(x, Haar, s)
I = c.coeffs
h = Haar(s)
#%%
I2 = cwt_arbitrary_shifts(x, h, dv=0.2)

# %%
c.compute_ews(0.01, n_iter=10000)
# %%
c.view_ews()
# %%
from plotting import view

# %%
view(I, s, 1)
# %%
view(I2, s, 1)
# %%
I3 = cwt_arbitrary_shifts(x, h, dv=1)
# %%
view(I3, s, 1)
# %%
