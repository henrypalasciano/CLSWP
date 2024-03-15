# %%
import numpy as np
from CLSWP_Object import CLSWP
from wavelets import Haar

x = np.random.randn(3000)
s = np.linspace(2, 200, 199)

# %%
c = CLSWP(x, Haar, s)


