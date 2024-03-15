# %%
import numpy as np
from CLSWP_Object import CLSWP
from wavelet_functions import Haar
import time

x = np.random.randn(3000)
s = np.linspace(2, 200, 199)

# %%
s = np.linspace(2, 300, 299)
c = CLSWP(x, Haar, s)

#delta = s[1] - s[0]
# Compute the raw wavelets periodogram and rescale by delta
#I = c.coeffs ** 2 / delta
A_h = c.A

e = np.real(np.linalg.eig(A_h)[0][0])
print(e)

I = c.coeffs ** 2

# %%


def daub_inv_iter_asym(y, A, mu, n_iter):
    
    x = np.random.uniform(0, 1, np.shape(y))
    x = np.zeros_like(y) + 0.5
    e = np.real(np.linalg.eig(A)[0][0])
    A = A / e
    A_y = A @ y - mu / 2
    A_2 = A @ A
    
    for i in range(n_iter):
        x += A_y - A_2 @ x
        x *= (x > 0)
        
    return x / e

def daub_inv_iter_asym2(y, A, mu, n_iter):
    
    x = np.random.uniform(0, 1, np.shape(y))
    x = np.zeros_like(y) + 0.5
    e = np.real(np.linalg.eig(A)[0][0])
    A = A / e
    A_y = A @ y
    A_2 = A @ A
    
    for i in range(n_iter):
        x = thr_asym(x + A_y - A_2 @ x, mu)
        
    return x / e

def thr_asym(x, mu):
    
    x[x < mu / 2] = 0 
    x[x >= mu / 2] -= mu / 2
    
    return x

def daub_inv_iter_asym3(y, A, mu, n_iter):
    
    x = np.random.uniform(0, 1, np.shape(y))
    x = np.zeros_like(y) + 0.5
    e = np.real(np.linalg.eig(A)[0][0])
    A = A / e
    A_y = A @ y - mu / 2
    A_2 = A @ A
    
    for i in range(n_iter):
        x = np.maximum(x + A_y - A_2 @ x, 0)
        
    return x / e

t1 = time.time()
a = daub_inv_iter_asym(I, A_h, 0.01, 10000)
t2 = time.time()
a2 = daub_inv_iter_asym2(I, A_h, 0.01, 10000)
t3 = time.time()
a3 = daub_inv_iter_asym3(I, A_h, 0.01, 10000)
t4 = time.time()
print(t2 - t1)
print(t3 - t2)
print(t4 - t3)
print(np.mean(np.abs(a-a2)), np.mean(np.abs(a2-a3)))

# %%
np.maximum(np.array([1,-1,3]), 0)

# %%
