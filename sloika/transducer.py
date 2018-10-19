import numpy as np
from sloika import viterbi_helpers
from sloika.config import sloika_dtype

_NEG_LARGE = -50000.0
_STAY = 0


def argmax(*args):
    res = max(enumerate(args), key=lambda x: x[1])
    return res


def map_to_sequence(trans, sequence, slip=None, prior_initial=None, prior_final=None, log=True):
    """  Find Viterbi path through sequence for transducer

    :param trans: A 2D :class:`nd.array` Transducer to be mapped
    :param sequence: A 1D :class:`nd.array` Sequence of bases to be mapped against
    :param slip: slip penalty (in log-space)
    :param prior_initial: A 1D :class:`nd.array` containing prior over initial position
    :param prior_final: A 1D :class:`nd.array` containing prior over final position
    :param log: Transducer is log-scaled

    :returns: Tuple containing score for path and array containing path
    """
    assert slip is None or slip >= 0.0, 'Slip penalty should be non-negative'
    slip = np.float32(slip)
    nev = len(trans)
    npos = len(sequence)
    ltrans = trans if log else np.log(trans)

    # Matrix for Viterbi traceback of path
    vmat = np.zeros((nev, npos), dtype=np.int32)
    # Vectors for current and previous score
    pscore = np.zeros(npos, dtype=sloika_dtype)
    cscore = np.zeros(npos, dtype=sloika_dtype)

    # Initialisation
    if prior_initial is not None:
        pscore += prior_initial
    pscore += np.fmax(ltrans[0][sequence], ltrans[0][_STAY])

    # Main loop
    for i in range(1, nev):
        ctrans = ltrans[i]
        # Stay
        vmat[i] = np.arange(0, npos)
        cscore = pscore + ctrans[_STAY]
        # Step
        step_score = pscore[:-1] + ctrans[sequence[1:]]
        move = np.where(step_score > cscore[1:])[0]
        cscore[move + 1] = step_score[move]
        vmat[i][move + 1] = move
        # Slip
        if slip is not None:
            from_score, from_pos = viterbi_helpers.slip_update(pscore, slip)
            from_score += ctrans[sequence]
            vmat[i] = np.where(from_score <= cscore, vmat[i], from_pos)
            cscore = np.where(from_score <= cscore, cscore, from_score)

        pscore, cscore = cscore, pscore

    if prior_final is not None:
        pscore += prior_final

    # Viterbi traceback
    path = np.empty(nev, dtype=np.int32)
    path[0] = np.argmax(pscore)
    max_score = pscore[path[0]]
    for i in range(1, nev):
        path[i] = vmat[nev - i][path[i - 1]]

    return max_score, path[::-1]
