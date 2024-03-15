import numpy as np
from wavelets import Wavelet

# ================================================================================================
# Continuous Wavelet Transform
# ================================================================================================

def cwt(x: np.ndarray, Wavelet: Wavelet, sampling_rate: int = 1, bc: str = "periodic",
        missing_data: bool = False, irregularly_spaced: bool = False, 
        times: np.array = None, min_spacing: float = None) -> np.ndarray:
    
    if missing_data:
        x = reweight(x)
        
    if irregularly_spaced:
        x = spacing_function(x, times, min_spacing)
    
    if bc == "periodic":
        return cwt_periodic(x, Wavelet, sampling_rate)
    
    elif bc == "symmetric":
        return cwt_symmetric(x, Wavelet, sampling_rate)
    
    elif bc == "zero":
        return cwt_zero(x, Wavelet, sampling_rate)
    
    elif bc == "constant":
        return cwt_constant(x, Wavelet, sampling_rate)
    
    else:
        raise TypeError("""Boundary condition {bc} not implemented. Available options:\n 
                        symmetric, periodic, constant or zero.""".format(bc=bc))
    


# ================================================================================================
# Continuous Wavelet Transform - Zero Padding
# ================================================================================================

def cwt_zero(x, Wavelet, sampling_rate=1, omega=0):
    
    scales = Wavelet.scales
    s = len(scales)
    n = len(x)
    c = np.empty([s,n])
    
    delta = 1 / sampling_rate
    a = int(np.ceil(n / 2))
    b = int(np.floor(n / 2))
    
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m), scale)
        if m > n:
            k1 = np.round(m / 2 - a).astype(int)
            k2 = np.round(m / 2 + b).astype(int)
            W_i = W_i[k1:k2]
        
        c_i = np.convolve(W_i, x, "same")
        c[i] = c_i
    
    return c * delta


# ================================================================================================
# Continuou Wavelet Transform - Period Boundary
# ================================================================================================


def cwt_periodic(x, Wavelet, sampling_rate=1, omega=0):
    
    scales = Wavelet.scales
    s = len(scales)
    n = len(x)
    c = np.empty([s,n])
    
    delta = 1 / sampling_rate
    
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        k = (m - 1) / 2
        if k != 0:
            a = int(np.ceil(k))
            b = int(np.floor(k))
            x_i = x.take(range(-a, n+b), mode="wrap")
        else: 
            x_i = x
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m), scale)
        c_i = np.convolve(W_i, x_i, "valid")
        c[i] = c_i
    
    return c * delta

# ================================================================================================
# Continuou Wavelet Transform - Symmetric Boundary
# ================================================================================================


def cwt_symmetric(x, Wavelet, sampling_rate=1, omega=0):
    
    scales = Wavelet.scales
    s = len(scales)
    n = len(x)
    c = np.empty([s,n])
    
    delta = 1 / sampling_rate
    
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    x_original = x
    x = np.hstack([x[::-1], x, x[::-1]])
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        if m > 0:
            k = (m - 1) / 2
            if k != 0:
                a = int(np.ceil(k))
                b = int(np.floor(k))
                x_i = x.take(range(n-a, 2*n+b), mode="wrap")
            else: 
                x_i = x_original
            W_i = Wavelet.wavelet(np.linspace(lb, ub, m) + omega, scale)
            c_i = np.convolve(W_i, x_i, "valid")
            c[i] = c_i
            
    return c * delta


# ================================================================================================
# Continuou Wavelet Transform - Constant Boundary
# ================================================================================================


def cwt_constant(x, Wavelet, sampling_rate=1, omega=0):
    
    scales = Wavelet.scales
    s = len(scales)
    n = len(x)
    c = np.empty([s,n])
    
    delta = 1 / sampling_rate
    
    lower, upper, n_points = Wavelet.wavelet_filter(sampling_rate)
    x_l = x[0]
    x_r = x[-1]
    
    for i,scale in enumerate(scales):
        lb, ub, m = lower[i], upper[i], n_points[i]
        k = (m - 1) / 2
        if k != 0:
            a = int(np.ceil(k))
            b = int(np.floor(k))
            x_i = np.hstack([np.array([x_l] * a), x, np.array([x_r] * b)])
        else: 
            x_i = x
        W_i = Wavelet.wavelet(np.linspace(lb, ub, m), scale)
        c_i = np.convolve(W_i, x_i, "valid")
        c[i] = c_i
    
    return c * delta



# ================================================================================================
# Reweighting and Spacing Functions
# ================================================================================================

def reweight(x):
    
    is_nan = np.isnan(x)
    while(is_nan[0]):
        x = x[1:]
        is_nan = is_nan[1:]
    
    idx = np.where(is_nan)[0]
    weights = np.ones_like(x)
    for i,j in enumerate(idx):
        if j - 1 not in idx:
            if j + 1 not in idx:
                weights[j - 1] += 1/2
                weights[j + 1] += 1/2
            else:
                mj = j
                count = 1
        else:
            count += 1
            if j + 1 not in idx:
                weights[mj - 1] += count/2
                weights[j + 1] += count/2
    
    x[idx] = 0
    x = x * weights
    
    return x


def spacing_function(x, times, min_spacing):
    
    times = ((times - np.min(times)) / min_spacing).astype(int)
    t = np.linspace(0, np.max(times), int(np.max(times)) + 1)
    y = np.zeros_like(t)
    y[:] = np.nan
    y[times] = x

    return reweight(y)




# ================================================================================================
# Continuous Wavelet Transform with Irregularly Spaced Data
# ================================================================================================

def cwt_periodic_i(x, t, Wavelet, dv=None):
    
    s = len(Wavelet.scales)
    n = len(x)
    
    t = t - t[0]
    
    if dv == None:
        dv = np.mean(t[1:] - t[:-1])
        
    m = int((np.max(t) - np.min(t)) / dv)
    c = np.zeros([s, m])
    
    sampling_rate = t[-1] / np.mean(t[1:] - t[:-1])
    
    N = int(min(np.max(Wavelet.scales) * sampling_rate, n//2))
    x = np.hstack([x[-N:-1], x, x[1:N]])
    t = np.hstack([t[-N:-1] - t[-1], t, t[1:N] + t[-1]])
    
    dt = np.hstack([t[1], (t[2:] - t[:-2]) / 2, t[-1] - t[-2]])
    
    xdt = x * dt
    
    for i in range(m):
        if i%100 == 0:
            print(".", end=" ")
        W = Wavelet.wavelet(i * dv - t)
        c_i = W @ xdt
        c[:,i] = c_i
    
    return c
    
 
# ================================================================================================
# Continuous Wavelet Transform with Arbitrary Shifts
# ================================================================================================

def cwt_arbitrary_shifts(x: np.ndarray, Wavelet: Wavelet, sampling_rate: int = 1, dv=1,
                         bc: str = "periodic") -> np.ndarray:
    
    delta = 1 / sampling_rate
    k = int(delta // dv)
    dv = delta / k
    
    n = len(x)
    s = len(Wavelet.scales)
    c = np.empty([s, k * n])
    x_n = np.array([range(n * k)])
    
        
    if bc == "periodic":
        for i in range(k):
            ind = (x_n % k == i)[0]
            c[:, ind] = cwt_periodic(x, Wavelet, sampling_rate=sampling_rate, omega=dv * i)
        return c
    
    elif bc == "symmetric":
        for i in range(k):
            ind = (x_n % k == i)[0]
            c[:, ind] = cwt_symmetric(x, Wavelet, sampling_rate=sampling_rate, omega=dv * i)
        return c
    
    elif bc == "zero":
        for i in range(k):
            ind = (x_n % k == i)[0]
            c[:, ind] = cwt_zero(x, Wavelet, sampling_rate=sampling_rate, omega=dv * i)
        return c
    
    elif bc == "constant":
        raise TypeError("Not implemented yet")
    
    else:
        raise TypeError("""Boundary condition {bc} not implemented. Available options:\n 
                        symmetric, periodic, constant or zero.""".format(bc=bc))





def cwt_periodic_s(x, Wavelet, sampling_rate=1, dv=1):
    
    delta = 1 / sampling_rate
    k = int(delta // dv)
    dv = delta / k
    
    n = len(x)
    s = len(Wavelet.scales)
    c = np.empty([s, k * n])
    x_n = np.array([range(n * k)])
    
    for i in range(k):
        ind = (x_n % k == i)[0]
        c[:, ind] = cwt_periodic(x, Wavelet, sampling_rate=sampling_rate, omega=dv * k)
    
    return c

