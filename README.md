# Continuous Time Locally Stationary Wavelet Processes

This repository provides a **Python implementation** of **Continuous Time Locally Stationary Wavelet Processes**,  
as introduced in the Biometrika paper:  
📄 **[Continuous Time Locally Stationary Wavelet Processes](https://doi.org/10.1093/biomet/asaf015)**  

This work extends the **discrete-time** locally stationary wavelet processes of  [Nason et al. (2000)](https://doi.org/10.1111/1467-9868.00231) to the **continuous-time** setting.

---

## Repository Structure

The repository is organized as follows:

```plaintext
📂 CLSWP/
 ┣ 📂 methodology/       # Python implementation of Continuous Time Locally Stationary Wavelet Processes (CLSWP)
 ┃ ┣ 📜 CLSWP.py         # Defines CLSWP classes, including those for handling missing and irregular time series data
 ┃ ┣ 📜 wavelets.py      # Implements predefined wavelet objects (Haar, Ricker, Morlet, Shannon) with support for custom wavelets
 ┃ ┣ 📜 cwt.py           # Computes the continuous wavelet transform (CWT)
 ┃ ┣ 📜 ews.py           # Computes the evolutionary wavelet spectrum (EWS) from the raw wavelet periodogram
 ┃ ┣ 📜 local_acf.py     # Computes local autocovariance and autocorrelation from the evolutionary wavelet spectrum
 ┃ ┣ 📜 smoothing.py     # Implementation of wavelet-based smoothing for the raw wavelet periodogram
 ┃ ┣ 📜 plotting.py      # Visualization utilities for CLSWP analysis
 ┣ 📜 README.md          # Project documentation
 ┣ 📜 requirements.txt   # List of dependencies
```


---

## Example Usage

### Regularly Spaced Observations

```python
import numpy as np
from haar_MA import haar_MA
from CLSWP import CLSWP
from wavelets import Haar

# Generate a Haar wavelet process
ts = haar_MA(10, 1, 1000)

# Select scales of interest
scales = np.linspace(0.05, 5, 100)

# Initialize the process
lsw = CLSWP(ts, Haar, scales, sampling_rate=100, bc="symmetric")

# Smooth raw wavelet periodogram
lsw.smooth_rwp(smooth_wav="db3")

# Compute the evolutionary wavelet spectrum
lsw.compute_ews(N=1000)

# Compute local autocovariance and autocorrelation
tau = np.linspace(0, 5, 101)
lsw.compute_local_acf(tau)

# Visualize the results
lsw.view("Evolutionary Wavelet Spectrum")
```

### Irregularly Spaced Observations

```python
import numpy as np
from CLSWP import CLSWPIrregularlySpacedData
from wavelets import Haar

# Generate an irregular time series
observed = np.sort(np.random.choice(1000, 750))
x = haar_MA(10, 1, 100)[observed]
times = np.arange(0, 10, 0.01)[observed]

# Initialize the process for irregularly spaced data
lsw = CLSWPIrregularlySpacedData(x, Haar, scales, times, sampling_rate=100, bc="symmetric", keep_all=False)

# Compute wavelet-based estimates
lsw.compute_ews(N=1000)
lsw.compute_local_acf(tau)

# Visualize results
lsw.view("Evolutionary Wavelet Spectrum")
```

---

## Citing this work

If you use **Continuous Time Locally Stationary Wavelet Processes** in your research, please cite the following paper:  


```bibtex
@article{CLSWP,
  author    = {Palasciano, H. A. and Knight, M. I. and Nason G. P.},
  title     = {Continuous Time Locally Stationary Wavelet Processes},
  journal   = {Biometrika},
  year      = {2025},
  doi       = {10.1093/biomet/asaf015},
}
```

---

## Contact  

**Henry Antonio Palasciano**  
📧 Email: [henry.palasciano17@imperial.ac.uk](mailto:henry.palasciano17@imperial.ac.uk)

**Marina I. Knight**  
📧 Email: [marina.knight@york.ac.uk](mailto:marina.knight@york.ac.uk)  

**Guy P. Nason**  
📧 Email: [gnason@imperial.ac.uk](mailto:gnason@imperial.ac.uk)  