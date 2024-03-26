# Continuous Time Locally Stationary Wavelet Processes
## Author: Henry Antonio Palasciano

Functions to compute the continuous wavelet transform, evolutionary wavelet spectrum and local autocovariance and autocorrelation for a given time series, using the methods developed in https://arxiv.org/abs/2310.12788.

### Contents:
* wavelets.py - wavelet objects. Available wavelets: Haar, Ricker, Shannon;
* cwt.py - functions for computing the continuous wavelet transform for a range of boundary conditions;
* ews.py - functions for computing an estimate of the evolutionary wavelet spectrum;
* local_acf.py - functions for computing estimates of the local autocovariance and autocorrelation;
* plotting.py - functions for plotting estimates and quantities of interest;
* smoothing.py - functions for smoothing on the raw wavelet periodogram;
* CLSWP - classes for creating an instance of the continuous locally staionary wavelet process, for which one can compute the evolutionary wavelet spectrum and local autocovariance. Additional classes for time series with missing or irregularly spaced observations also available.

### Example 1:
Continuous time locally stationary wavelet processes for time series with regularly spaced observations.
```python
import numpy as np
# Sample time series
ts = np.random.randn(1000)
```
Create an instance of the CLSWP class for the time series of interest using Haar Wavelets
```python
from CLSWP import CLSWP
from wavelets import Haar
# Select scales of interest
scales = np.linspace(2, 200, 100)
# Initialise the locally stationary wavelet process
lsw = CLSWP(ts, Haar, scales, sampling_rate=1, bc="symmetric")
```
Compute estimates of the evolutionary wavelet spectrum and local acf
```python
# Estimate evolutionary wavelet spectrum
lsw.compute_ews(mu=0.01, n_iter=1000)
# Lags of interest
tau = np.linspace(0, 50, 51)
# Compute local autocovariance and autocorralation from the estimate of the evolutionary wavelet spectrum
lsw.compute_local_acf(tau)
```
Plot any quantities of interest
```python
# Plot the evolutionary wavelet spectrum
lsw.view("Evolutionary Wavelet Spectrum")
```

### Example 2:
Continuous time locally stationary wavelet processes for time series with irregularly spaced observations.
```python
import numpy as np
# Sample time series
ts = np.random.randn(1000)
# Irregularly spaced observation times
times = np.sort(np.random.choice(np.linspace(0, 1000, 4001), 1000, replace=False))
```
Create an instance of the CLSWP class for irregularly spaced data for the time series of interest using Haar Wavelets
```python
from CLSWP import CLSWPIrregularlySpacedData
from wavelets import Haar
# Select scales of interest
scales = np.linspace(2, 200, 100)
# Initialise the locally stationary wavelet process for irregularly spaced data.
lsw = CLSWPIrregularlySpacedData(ts, Haar, scales, times, sampling_rate=0.25, bc="symmetric", keep_all=False)
```
The sampling rate determines the minimum distance between any two observations when mapped to a regularly spaced grid. It can also be interpreted as the shift from one location to the next while computing the cwt. The keep_all parameter determines whether the user would like to estimate the evolutionary wavelet spectrum at all locations or only where the data was originally present. Note that once the cwt is computed, the ews is independent of location.  
Compute estimates of the evolutionary wavelet spectrum and local acf
```python
# Estimate evolutionary wavelet spectrum
lsw.compute_ews(mu=0.01, n_iter=1000)
# Lags of interest
tau = np.linspace(0, 50, 51)
# Compute local autocovariance and autocorralation from the estimate of the evolutionary wavelet spectrum
lsw.compute_local_acf(tau)
```
Plot any quantities of interest
```python
# Plot the evolutionary wavelet spectrum
lsw.view("Evolutionary Wavelet Spectrum")
```