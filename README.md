# Continuous Time Locally Stationary Wavelet Processes
## Author: Henry Antonio Palasciano

Functions to compute the continuous wavelet transform and evolutionary wavelet transform for a given time series.

Contents:
* wavelets.py - wavelet objects. Available wavelets: Haar, Ricker, Shannon;
* cwt.py - functions for computing the continuous wavelet transform for a range of boundary conditions;
* ews.py - functions for computing an estimate of the evolutionary wavelet spectrum;
* local_acf.py - functions for computing estimates of the local autocovariance and autocorrelation;
* plotting.py - functions for plotting estimates and quantities of interest;
* smoothing.py - functions for smoothing on the raw wavelet periodogram;
* CLSWP - classes for creating an instance of the continuous locally staionary wavelet process, for which one can compute the evolutionary wavelet spectrum and local autocovariance. Additional classes for time series with missing or irregularly spaced observations also available.

Link to paper: https://arxiv.org/abs/2310.12788