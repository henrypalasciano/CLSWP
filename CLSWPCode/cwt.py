import numpy as np
from wavelets import Wavelet

# ================================================================================================
# Continuous Wavelet Transform
# ================================================================================================

def cwt(x: np.ndarray, Wavelet: Wavelet, sampling_rate: float = 1, bc: str = "periodic",
        missing_data: bool = False, irregularly_spaced: bool = False, 
        times: np.array = None, min_spacing: float = None, alpha: float = 0) -> np.ndarray:
    """
    Computes the Continuous Wavelet Transform (CWT) of a given time series.

    Args:
        x (np.ndarray): The input time series.
        Wavelet (Wavelet): The wavelet to be used for the transform. This is an instance of the Wavelet class, with the desired scales.
        sampling_rate (int): The sampling rate of the signal. Default is 1.
        bc (str): The boundary condition to be used. Available options are "periodic", "symmetric", "zero", and "constant". Default is "periodic".
        missing_data (bool): Whether the time series contains missing data. Default is False. Set to False if irregularly_spaced is True.
        irregularly_spaced (bool): Whether the time series is irregularly spaced. Default is False. Set to False if missing_data is True.
        times (np.array): The corresponding times for the series. Required if irregularly_spaced is True.
        min_spacing (float): The minimum spacing between any two observations. Required if irregularly_spaced is True.
        alpha (float): The shift parameter. Default is 0.

    Returns:
        np.ndarray: The CWT of the input time series.
    """
    if missing_data:
        x = reweight(x)
    elif irregularly_spaced:
        x = spacing_function(x, times, min_spacing)
    
    if bc == "periodic":
        return cwt_periodic_or_constant(x, Wavelet, mode="wrap", sampling_rate=sampling_rate, alpha=alpha)
    elif bc == "symmetric":
        return cwt_symmetric(x, Wavelet, sampling_rate=sampling_rate, alpha=alpha)
    elif bc == "zero":
        return cwt_zero(x, Wavelet, sampling_rate=sampling_rate, alpha=alpha)
    elif bc == "constant":
        return cwt_periodic_or_constant(x, Wavelet, mode="clip", sampling_rate=sampling_rate, alpha=alpha)
    else:
        raise TypeError("""Invalid boundary condition: {bc}. Available options:\n 
                        symmetric, periodic, constant or zero.""".format(bc=bc))
    
# ================================================================================================
# Continuous Wavelet Transform - Periodic or Constant Boundary
# ================================================================================================


def cwt_periodic_or_constant(x: np.ndarray, Wavelet: Wavelet, mode: str, sampling_rate: float = 1, alpha: float = 0) -> np.ndarray:
    """
    Computes the continuous wavelet transform (CWT) using periodic or constant boundary conditions.

    Args:
        x (np.ndarray): The input signal.
        Wavelet (np.ndarray): The wavelet to be used for the CWT.
        mode (str): {"wrap", "clip"}. Specifies how out-of-bounds indices will behave. "wrap" for periodic and "clip" for constant.
        sampling_rate (float, optional): The sampling rate of the input signal. Defaults to 1.
        alpha (float, optional): The shift parameter. Defaults to 0.

    Returns:
        np.ndarray: The CWT coefficients.
    """
    # Extract scales
    scales = Wavelet.scales
    # Initialise array of coefficients
    n = len(x)
    c = np.zeros([len(scales), n])
    # Points at which to sample the wavelet for each scale
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        # If the number of points is less than or equal to 1, skip the iteration
        if m <= 1:
            continue
        # Enforce boundary condtions
        k = (m - 1) / 2
        if k != 0:
            a = int(np.ceil(k))
            b = int(np.floor(k))
            x_i = x.take(range(-a, n+b), mode=mode)
        else: 
            x_i = x
        # Sample wavelet and compute CWT
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m) + alpha, scale)
        c[i] = np.convolve(W_i, x_i, "valid")
    
    return c / sampling_rate

# ================================================================================================
# Continuous Wavelet Transform - Symmetric Boundary
# ================================================================================================


def cwt_symmetric(x: np.ndarray, Wavelet: Wavelet, sampling_rate: float = 1, alpha: float = 0) -> np.ndarray:
    """
    Computes the continuous wavelet transform (CWT) using symmetric boundary conditions.

    Args:
        x (np.ndarray): The input signal.
        Wavelet (np.ndarray): The wavelet to be used for the CWT.
        sampling_rate (float, optional): The sampling rate of the input signal. Defaults to 1.
        alpha (float, optional): The shift parameter. Defaults to 0.

    Returns:
        np.ndarray: The CWT coefficients.
    """
    # Extract scales
    scales = Wavelet.scales
    # Initialise array of coefficients
    n = len(x)
    c = np.empty([len(scales), n])
    # Points at which to sample the wavelet for each scale
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    # Extend the time series symmetrically
    x = np.hstack([x, x[::-1]])
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        # If the number of points is less than or equal to 1, skip the iteration
        if m <= 1:
            continue
        # Enforce symmetric boundary conditions
        k = (m - 1) / 2
        if k != 0:
            a = int(np.ceil(k))
            b = int(np.floor(k))
            x_i = x.take(range(-a, n+b), mode="wrap")
        else: 
            x_i = x[:n]
        # Sample wavelet and compute CWT
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m) + alpha, scale)
        c[i] = np.convolve(W_i, x_i, "valid")
            
    return c / sampling_rate

# ================================================================================================
# Continuous Wavelet Transform - Zero Padding
# ================================================================================================

def cwt_zero(x: np.ndarray, Wavelet: Wavelet, sampling_rate: float = 1, alpha: float = 0) -> np.ndarray:
    """
    Computes the continuous wavelet transform (CWT) using zero padding at the boundaries.

    Args:
        x (np.ndarray): The input signal.
        Wavelet (np.ndarray): The wavelet to be used for the CWT.
        sampling_rate (float, optional): The sampling rate of the input signal. Defaults to 1.
        alpha (float, optional): The shift parameter. Defaults to 0.

    Returns:
        np.ndarray: The CWT coefficients.
    """
    # Extract scales
    scales = Wavelet.scales
    # Initialise array of coefficients
    n = len(x)
    c = np.empty([len(scales), n])
    # 
    a = int(np.ceil(n / 2))
    b = int(np.floor(n / 2))
    # Points at which to sample the wavelet for each scale
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        # Sample wavelet
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m) + alpha, scale)
        # Trim wavelet if its wider than the time series
        if m > n:
            k1 = np.round(m / 2 - a).astype(int)
            k2 = np.round(m / 2 + b).astype(int)
            W_i = W_i[k1:k2]
        # Compute CWT
        c[i] = np.convolve(W_i, x, "same")
    
    return c / sampling_rate

# ================================================================================================
# Reweighting and Spacing Functions
# ================================================================================================

def reweight(x):
    """
    Reweights the values in the input array based on the number of NaNs nearby.
    This function is used for computing integrals (specifically for the CWT) when values are missing. It assigns weights to each value in the array based on the number of NaNs nearby. NaNs are then replaced by zeros, allowing for a more efficient CWT computation.
    
    Args:
        x (ndarray): Input array containing NaNs.
    
    Returns:
        np.ndarray: The reweighted array, with NaNs replaced by zeros.
    """
    # Find NaNs
    idx = ~np.isnan(x)
    x[~idx] = 0
    # Remove leading and trailing NaNs
    x = np.trim_zeros(x)
    idx = np.trim_zeros(idx)
    # Initialise weights
    weights = idx.copy().astype(float)
    t = np.arange(0, len(x))[idx]
    # Adjust weights based on the spacing between valid observations
    weights[idx] += np.convolve((t[1:] - t[:-1] - 1) / 2, np.array([1, 1]))
    # Reweight the data accordingly
    return x * weights

def spacing_function(x, times, min_spacing):
    """
    Converts irregularly spaced data to regularly spaced data with NaNs in the missing points. Then reweights the data to compute the integrals for the CWT.

    Args:
        x (np.ndarray): The input time series.
        times (np.ndarray): The corresponding times.
        min_spacing (float): The minimum spacing between any two observations.

    Returns:
        np.ndarray: The regularly spaced data, reweighted based on the distance to neighbouring points and with zeros at the missing locations.
    """
    # Rescale times, so that the first observation is at time 0 and the minimum spacing between any two observations is one
    times = ((times - np.min(times)) / min_spacing).astype(int)
    # Create a new array with NaNs at the missing locations
    y = np.ones(times[-1] + 1) * np.nan
    y[times] = x
    # Reweight the time series
    return reweight(y)
 
# ================================================================================================
# Continuous Wavelet Transform with Arbitrary Shifts
# ================================================================================================

def cwt_arbitrary_shifts(x: np.ndarray, Wavelet: Wavelet, sampling_rate: float = 1, dv: float = 1,
                         bc: str = "periodic", missing_data: bool = False, irregularly_spaced: bool = False, 
                         times: np.array = None, min_spacing: float = None) -> np.ndarray:
    """
    Computes the continuous wavelet transform for arbitrary shifts of the wavelet, rather than restricting this to one.

    Args:
        x (np.ndarray): The input time series.
        Wavelet (Wavelet): The wavelet to be used for the transform. This is an instance of the Wavelet class, with the desired scales.
        sampling_rate (float): The sampling rate of the signal. Default is 1.
        dv (float): The step size for the arbitrary shifts. Default is 1.
        bc (str): The boundary condition to be used. Available options are "periodic", "symmetric", "zero", and "constant". Default is "periodic".
        missing_data (bool): Whether the time series contains missing data. Default is False. Set to False if irregularly_spaced is True.
        irregularly_spaced (bool): Whether the time series is irregularly spaced. Default is False. Set to False if missing_data is True.
        times (np.array): The corresponding times for the series. Required if irregularly_spaced is True.
        min_spacing (float): The minimum spacing between any two observations. Required if irregularly_spaced is True.

    Returns:
        np.ndarray: The continuous wavelet transform of the input time series.
    """
    if missing_data:
        x = reweight(x)
    elif irregularly_spaced:
        x = spacing_function(x, times, min_spacing)
    # Calculate the number of shifts between any two consecutive observations
    k = sampling_rate / dv
    if k == 1: # Return the standard CWT
        return cwt(x, Wavelet, sampling_rate=sampling_rate, bc=bc, missing_data=False,
                   irregularly_spaced=False)
    elif k > 1: # Multiple shift between observations, so calculate the CWT for each shift and combine
        k = int(k)
        c = np.zeros([len(Wavelet.scales), k * len(x)])
        for i in range(k):
            c[:, i::k] = cwt(x, Wavelet, sampling_rate=sampling_rate, bc=bc, missing_data=False,
                                    irregularly_spaced=False, alpha=i/k)
        return c
    elif k < 1: # Shift is greater than distance between observations, so calculate the standard CWT and downsample
        return cwt(x, Wavelet, sampling_rate=sampling_rate, bc=bc, missing_data=False, 
                   irregularly_spaced=False)[:, ::int(1/k)]
        
        