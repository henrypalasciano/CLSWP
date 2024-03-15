# %%
import numpy as np
from CLSWP_Object import CLSWP
from wavelet_functions import Haar
import time

x = np.random.randn(3000)
s = np.linspace(2, 200, 199)

# %%
c = CLSWP(x, Haar, s)


