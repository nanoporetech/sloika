import numpy as np


def med_mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
    with the standard deviation of a normal distribution
    (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate over
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: a tuple containing the median and MAD of the data
    """
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad


def mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and (by default)
    adjust by a factor for asymptotically normal consistency.

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
    with the standard deviation of a normal distribution
    (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate the median over.
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: the (scaled) MAD
    """
    _ , dmad = med_mad(data, factor=factor, axis=axis, keepdims=keepdims)
    return dmad


def studentise(x, axis=None):
    """  Studentise a numpy array along a given axis
    :param x: A :class:`ndaray`
    :param axis: axis over which to studentise

    :returns: A :class:`nd.array` with same shape as x
    """
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    s = np.where(s > 0.0, s, 1.0)
    return np.divide(x - m, s)
