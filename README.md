# Continuous Time Locally Stationary Wavelet Processes

📄 **"Continuous Time Locally Stationary Wavelet Processes"**\
🔗 **[Read the article in Biometrika](https://doi.org/10.1093/biomet/asaf015)**

---

## Overview



---

## Repository Structure

The repository is organized as follows:

```plaintext
📂 CLSWP/
 ┣ 📂 methodology/       # Core source code
 ┃ ┣ 📜 wavelets.py      # Defines wavelet objects (Haar, Ricker, Morlet, Shannon)
 ┃ ┣ 📜 cwt.py           # Computes the continuous wavelet transform (CWT)
 ┃ ┣ 📜 ews.py           # Computes the evolutionary wavelet spectrum (EWS)
 ┃ ┣ 📜 local_acf.py     # Computes local autocovariance and autocorrelation
 ┃ ┣ 📜 smoothing.py     # Functions for smoothing the raw wavelet periodogram
 ┃ ┣ 📜 plotting.py      # Visualization tools for results
 ┃ ┣ 📜 CLSWP.py         # Classes for creating CLSWP instances (supports irregular data)
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

## Citation

If you use this code in your research, please cite:

> **H. A. Palasciano, M. I. Knight and G. P. Nason** (2025). *Continuous Time Locally Stationary Wavelet Processes*. [Biometrika, asaf015](https://doi.org/10.1093/biomet/asaf015).

BibTeX entry:

```bibtex
@article{Palasciano2023,
  author    = {Henry Antonio Palasciano},
  title     = {Continuous Time Locally Stationary Wavelet Processes},
  journal   = {Biometrika},
  year      = {2023},
  doi       = {10.1093/biomet/asaf015},
}
```

---