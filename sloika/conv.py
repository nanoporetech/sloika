from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T
import theano.tensor.signal.pool as tp

PADDING_MODES = frozenset(['same', 'half', 'valid', 'full', 'same_left'])


def calculate_padding(mode, winlen):
    """Calculate padding amount for given convolution mode and window length

        Mode          Padding/Notes
        ----          -------------

        'same'        [(winlen - 1) // 2, winlen // 2]
                      As tensorflow's 'SAME' mode. Padding is as symmetric as
                      possible, with an extra 0 at the end when winlen is even.
                      Using this mode, convolution output size is independent of
                      window length.

        'half'        [winlen // 2, winlen // 2]
                      As Theano's 'half' mode.

        'valid'       [0, 0]
                      As Theano's 'valid' mode, and tensorflow's 'VALID' mode.

        'full'        [winlen - 1, winlen - 1]
                      As Theano's 'full' mode.

        'same_left'   [winlen // 2, (winlen - 1) // 2]
                      Like 'same' with extra 0 at the start when winlen is even.

        int           [int, int]

        int1, int2    [int1, int2]

    :param mode: str, int or (int, int)
        padding mode name, or integer(s) specifying padding amount
    :param winlen: convolution window length or pool size

    :returns: (padding to start, padding to end)
    """
    assert winlen > 0, "winlen must be positive"
    if isinstance(mode, int):
        return (mode, mode)
    if isinstance(mode, tuple):
        if map(type, mode) == [int, int]:
            return mode

    assert mode in PADDING_MODES, 'Padding mode "{}" not supported'.format(mode)
    if mode == "same":
        return ((winlen - 1) // 2, winlen // 2)
    if mode == "half":
        return (winlen // 2, winlen // 2)
    if mode == "valid":
        return (0, 0)
    if mode == "full":
        return (winlen - 1, winlen - 1)
    if mode == "same_left":
        return (winlen // 2, (winlen - 1) // 2)

    raise NotImplementedError("Padding mode case {} not dealt with".format(mode))


def pad_first(X, padding):
    """Pad first dimension of a tensor

    :param X: symbolic tensor to pad
    :param padding: tuple of ints (start padding, end padding)
    """
    assert len(padding) == 2, "Padding should be (int, int), got {!r}".format(padding)
    pad = T.shape_padleft(T.zeros(X.shape[1:]))
    pad_start = T.repeat(pad, padding[0], axis=0)
    pad_end = T.repeat(pad, padding[1], axis=0)
    X_pad = T.concatenate([pad_start, X, pad_end], 0)
    return X_pad


def bf1t(X):
    """Transpose from [time, batch, features] to [batch, features, 1, time]"""
    return T.shape_padaxis(X.transpose((1, 2, 0)), 2)


def tbf(X):
    """Tranpose from [batch, features, 1, time] to [time, batch, features]"""
    return X.transpose((3, 0, 1, 2)).flatten(3)


def conv_1d(X, W, stride=1, padding=(0, 0)):
    """Symbolic 1d convolution over first dimension

    This is a wrapper around theano.tensor.nnet.conv2d that expects an input
    of shape [time, batch, features] and performs a convolution over the
    time dimension.

    :param X: input of shape [time, batch, input_features]
    :param W: a filter of shape [out_features, in_features, winlen]
    :param stride: the rate of downsampling
    :param padding: (int, int) padding for start and end of time axis

    Returns:
        A 3D tensor of shape [ceil((time + padding) / stride), batch, out_features]
    """

    X_pad = pad_first(X, padding)
    conv = T.nnet.conv2d(bf1t(X_pad), T.shape_padaxis(W, 2),
                         subsample=(1, stride), filter_flip=False)
    Y = tbf(conv)

    return Y


def pool_1d(X, pool_size, stride, padding=(0, 0)):
    """Symbolic 1d max pool over first dimension

    This is a wrapper around theano.tensor.signal.pool.pool_2d that expects an
    input of shape [time, batch, features] and performs max pooling over the
    time dimension.

    :param X: input tensorof shape [time, batch, features]
    :param size: length of pool
    :param stride: level of downsampling
    :param padding: (int, int) padding for start and end of time axis

    Returns:
        3D tensor of shape [ceil((time + padding) / stride), batch, features]
    """

    X_pad = pad_first(X, padding)
    pool = tp.pool_2d(bf1t(X_pad), (1, pool_size), st=(1, stride),
                      ignore_border=True)
    Y = tbf(pool)

    return Y
