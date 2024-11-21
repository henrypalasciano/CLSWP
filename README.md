# Continuous Time Locally Stationary Wavelet Processes
## Author: Henry Antonio Palasciano

Functions to compute the continuous wavelet transform, evolutionary wavelet spectrum and local autocovariance and autocorrelation for a given time series, using the methods developed in https://arxiv.org/abs/2310.12788.

### Contents:
* wavelets.py - wavelet objects. Available wavelets: Haar, Ricker, Shannon;
* cwt.py - functions for computing the continuous wavelet transform for a range of boundary conditions;
* ews.py - functions for computing an estimate of the evolutionary wavelet spectrum;
* local_acf.py - functions for computing estimates of the local autocovariance and autocorrelation;
* plotting.py - functions for plotting estimates and quantities of interest;
* smoothing.py - functions for smoothing the raw wavelet periodogram;
* CLSWP - classes for creating an instance of the continuous locally staionary wavelet process, for which one can compute the evolutionary wavelet spectrum and local autocovariance. Additional classes for time series with missing or irregularly spaced observations also available.

### Example 1:
Continuous time locally stationary wavelet processes for time series with regularly spaced observations.
```python
import numpy as np
from haar_MA import haar_MA
# Sample a stationary Haar process of length T = 10, order alpha = 1 and 1000 observations long
ts = haar_MA(10, 1, 100)
```
Create an instance of the CLSWP class for the time series of interest using Haar Wavelets
```python
from CLSWP import CLSWP
from wavelets import Haar
# Select scales of interest
scales = np.linspace(0.05, 5, 100)
# Initialise the locally stationary wavelet process
lsw = CLSWP(ts, Haar, scales, sampling_rate=100, bc="symmetric")
# Smooth raw wavelet periodogram
lsw.smooth_rwp(smooth_wav="db3")
```
Compute estimates of the evolutionary wavelet spectrum and local acf
```python
# Estimate evolutionary wavelet spectrum
lsw.compute_ews(N=1000)
# Lags of interest
tau = (np.linspace(0, 5, 101))
# Compute local autocovariance and autocorralation from the evolutionary wavelet spectrum
lsw.compute_local_acf(tau)
```
Plot any quantities of interest
```python
# Plot the evolutionary wavelet spectrum
lsw.view("Evolutionary Wavelet Spectrum")
```
Run the iterative soft-thresholding algorithm for another 1000 iterations on the finer scales
```python
# Update the estimate
lsw.compute_ews(N=1000, update=True, u_idx=50)
# Plot the evolutionary wavelet spectrum
lsw.view("Evolutionary Wavelet Spectrum")
```
### Example 2:
Continuous time locally stationary wavelet processes for time series with irregularly spaced observations.
```python
import numpy as np
# Irregularly spaced time series
observed = np.sort(np.random.choice(1000, 750))
x = haar_MA(10, 1, 100)[observed]
times = np.arange(0, 10, 0.01)[observed]
```
Create an instance of the CLSWP class for irregularly spaced data for the time series of interest using Haar Wavelets
```python
from CLSWP import CLSWPIrregularlySpacedData
from wavelets import Haar
# Select scales of interest
scales = np.linspace(0.05, 5, 100)
# Initialise the locally stationary wavelet process for irregularly spaced data.
lsw = CLSWPIrregularlySpacedData(ts, Haar, scales, times, sampling_rate=100, bc="symmetric", keep_all=False)
```
Compute estimates of the evolutionary wavelet spectrum and local acf
```python
# Estimate evolutionary wavelet spectrum
lsw.compute_ews(N=1000)
# Lags of interest
tau = (np.linspace(0, 5, 101))
# Compute local autocovariance and autocorralation from the estimate of the evolutionary wavelet spectrum
lsw.compute_local_acf(tau)
```
Plot any quantities of interest
```python
# Plot the evolutionary wavelet spectrum
lsw.view("Evolutionary Wavelet Spectrum")
```
