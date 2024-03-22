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
c.compute_ews(0.01, n_iter=100)
# %%
c.view_ews()
# %%
c.view_A()

# %%
import matplotlib.pyplot as plt
plt.imshow(I, aspect="equal")

# %%

