# Continuous Time Locally Stationary Wavelet Processes
## Author: Henry Antonio Palasciano

Functions to compute the continuous wavelet transform and evolutionary wavelet transform for a given time series.

Contents:
* wavelet_functions.py - wavelet function, autocorrelation and inner product operator. Available Wavelets: Haar, Ricker, Shannon;
* cwt_functions.py - functions for computing the continuous wavelet transform for different boundary conditions;
* ews_and_local_acf.py - functions for computing the evolutionary wavelet spectrum and local autocovariance and autocorrelation;
* CLSWP_Object - class for creating an instance of the continuous locally staionary wavelet process, for which one can compute the evolutionary wavelet spectrum and local autocovariance.

Link to paper: https://arxiv.org/abs/2310.12788