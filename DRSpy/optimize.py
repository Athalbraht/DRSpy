from DRSpy.analysis import np
from scipy.optimize import curve_fit

####### Func


def w_avg(x, w):
    """
    Weighted Average
    """
    x, w = np.array(x), np.array(w)
    return (x * w).sum() / w.sum()


def assymetry(c1, c2):
    """Assymetry func"""
    return (c1 - c2) / (c1 + c2)


def chspec(c1, c2):
    return np.sqrt(c1 * c2)


def chspec_ln(c1, c2):
    return np.log(c1 / c2)


def gamma_coff(x, L, p, q):
    """
    Gamma cofficient for time_fix() and sig_fix()
    """
    f1 = np.sqrt(p**2 + q**2 + (L - x) ** 2)
    f2 = np.sqrt(p**2 + q**2 + (L + x) ** 2)
    return f1 - f2


def time_fix(x, b, c=12.2, L=43.6, p=5.6, q=3):
    """
    Source position fix cofficient

    :param L: float
    :type L: float
    :param p: float, , defaults to
    :type p: float,
    :param q: float, , defaults to
    :type q: float,

    :returns: gamma cofficient
    :rtype: float
    """
    return 1 / c * gamma_coff(x, L, p, q) + b


def sig_fix(x, b, lm=117, L=43.6, p=5.6, q=3):
    return 2 / lm * gamma_coff(x, L, p, q) + b


####### Fit


def fit(func, X, Y, n=1000, xlim=False, maxfev=6500):
    """Fit engine"""
    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    if xlim:
        xmin, xmax = xlim[0], xlim[1]
    else:
        xmin, xmax = X.min(), X.max()
    params, pcov = curve_fit(func, X, Y, maxfev=maxfev)
    X_fit = np.linspace(xmin, xmax, n)
    Y_fit = func(X_fit, *params)
    return (X_fit, Y_fit, params, pcov)


def linear(x, a, b):
    """Linear function"""
    return a * x + b


def landau(x, E, S, N):
    """Landau Function"""
    return (
        1
        / np.sqrt(2 * np.pi)
        * np.exp(-((((x - E) / S) + np.exp(-((x - E) / S))) / 2))
        * N
    )


def decay_law(x, A, _lambda):
    """Decay law and light attenuation"""
    return A * np.exp(-_lambda * x)


def poly2(x, a, b, c):
    """Polynominal n2"""
    return c + b * x + a * x**2


def poly3(x, a, b, c, d):
    """Polynominal n3"""
    return d + c * x + b * x**2 + a * x**3


def inspector_calib(x, a):
    """Calibration function for InSpector-1000. f(x) = ax"""
    return a * x
