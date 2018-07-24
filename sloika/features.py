import numpy as np
from sloika.config import sloika_dtype
from sloika.maths import studentise


def from_events(ev, tag='scaled_', normalise=True, nanonet=False):
    """  Create a matrix of features from

    :param ev: A :class:`ndrecarray` with fields 'mean', 'stdv' and 'length'
    :param tag: Prefix of which fields to read
    :param normalise: Perform normalisation (Studentisation) of features.
    :param nanonet: Use Nanonet-like features

    :returns: A :class:`ndarray` with studentised features
    """
    nev = len(ev)
    features = np.zeros((nev, 4), dtype=sloika_dtype)
    features[:, 0] = ev[tag + 'mean']
    features[:, 1] = ev[tag + 'stdv']
    features[:, 2] = ev['length']
    #  Zero pad delta mean
    features[:, 3] = np.fabs(np.ediff1d(ev[tag + 'mean'], to_end=0))

    if normalise:
        features = studentise(features, axis=0)

    if nanonet:
        # Delta mean uncentred.
        features[:, 3] = np.ediff1d(ev[tag + 'mean'], to_end=0)
        features[:, 3] /= np.std(features[:, 3])

    return np.ascontiguousarray(features, dtype=sloika_dtype)
