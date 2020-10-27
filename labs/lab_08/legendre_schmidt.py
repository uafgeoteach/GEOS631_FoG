import numpy as np
from scipy.special import lpmv


def legendre_schmidt(n, x):
    """Compute associated Legendre functions with Schmidt normalization.

    Computes the associated Legendre functions of degree n and order
    m = 0, 1, ..., n evaluated for each element in x, with Schmidt
    normalization. Usage is identical to MATLAB's legendre() with 'sch'
    normalization.

    Args:
        n (int): Degree of Legendre function (must be positive)
        x (array_like): Input values

    Returns:
        Associated Legendre function values as a NumPy array. The shape
        of the returned array depends on the size of x:

        * If x is a 1D array, then an array of size (n+1)-by-len(x) is
          returned. The entry at index [m, i] is the associated Legendre
          function of degree n and order m evaluated at x(i).

        * In general, the output array has one more dimension than x and
          each element [m, i, j, k, ...] contains the associated Legendre
          function of degree n and order m evaluated at x[i, j, k, ...].
    """

    # Input checks and conversion
    if not (isinstance(n, int) and n >= 0):
        raise TypeError('n must be a positive integer')
    x = np.atleast_1d(x)
    if (x < -1).any() or (x > 1).any():
        raise ValueError('x must be in the range [-1, 1]')

    # Preallocate
    p_nm = np.empty((n + 1, *x.shape))

    # Compute while normalizing
    for m in range(p_nm.shape[0]):
        if m > 0:
            norm = ((-1) ** m) * np.sqrt((2 * np.math.factorial(n - m)) / np.math.factorial(n + m))
        else:
            norm = 1
        p_nm[m, ...] = lpmv(m, n, x) * norm  # lpmv() INCLUDES the Condon-Shortley phase factor

    return p_nm
