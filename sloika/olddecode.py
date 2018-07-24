import itertools
import numpy as np

_ETA = 1e-10
_BASES = ['A', 'C', 'G', 'T']
_DIBASES = [b1 + b2 for b1 in _BASES for b2 in _BASES]
_NSTEP = len(_BASES)
_NSKIP = _NSTEP ** 2
_STEP_FACTOR = np.log(_NSTEP)
_SKIP_FACTOR = np.log(_NSKIP)


def decode_profile(post, trans=None, log=False, slip=0.0):
    """  Viterbi-style decoding with per-event transition weights
    (profile)
    :param post: posterior probabilities of kmers by event.
    :param trans: A generator (e.g. a :class:`ndarray`) to produce
    per-transition log-scaled weights. None == no transition weights.
    :param log: Posterior probabilities are in log-space.
    """
    nstate = post.shape[1]
    lpost = post.copy()
    if not log:
        np.add(_ETA, lpost, lpost)
        np.log(lpost, lpost)

    if trans is None:
        trans = itertools.repeat(np.zeros(3))
    else:
        trans = np.copy(trans)
        trans[:, 1] -= _STEP_FACTOR
        trans[:, 2] -= _SKIP_FACTOR

    log_slip = np.log(_ETA + slip)

    pscore = lpost[0]
    trans_iter = trans.__iter__()
    for ev in range(1, len(post)):
        # Forward Viterbi iteration
        ev_trans = next(trans_iter)
        # Stay
        score = pscore + ev_trans[0]
        iscore = list(range(nstate))
        # Slip
        scoreNew = np.amax(pscore) + log_slip
        iscoreNew = np.argmax(pscore)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Step
        pscore = pscore.reshape((_NSTEP, -1))
        nrem = pscore.shape[1]
        scoreNew = np.repeat(np.amax(pscore, axis=0), _NSTEP) + ev_trans[1]
        iscoreNew = np.repeat(nrem * np.argmax(pscore, axis=0) + list(range(nrem)), _NSTEP)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Skip
        pscore = pscore.reshape((_NSKIP, -1))
        nrem = pscore.shape[1]
        scoreNew = np.repeat(np.amax(pscore, axis=0), _NSKIP) + ev_trans[2]
        iscoreNew = np.repeat(nrem * np.argmax(pscore, axis=0) + list(range(nrem)), _NSKIP)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Store
        lpost[ev - 1] = iscore
        pscore = score + lpost[ev]

    state_seq = np.zeros(len(post), dtype=int)
    state_seq[-1] = np.argmax(pscore)
    for ev in range(len(post), 1, -1):
        # Viterbi backtrace
        state_seq[ev - 2] = int(lpost[ev - 2][state_seq[ev - 1]])

    return np.amax(pscore), state_seq


def decode_transition(post, trans, log=False, slip=0.0):
    """  Viterbi-style decoding with weighted transitions
    :param post: posterior probabilities of kmers by event.
    :param trans: (log) penalty for [stay, step, skip]
    :param log: Posterior probabilities are in log-space.
    """
    return decode_profile(post, trans=itertools.repeat(trans), log=log, slip=slip)


def decode_simple(post, log=False, slip=0.0):
    """  Viterbi-style decoding with uniform transitions
    :param post: posterior probabilities of kmers by event.
    :param log: Posterior probabilities are in log-space.
    """
    return decode_profile(post, log=log, slip=slip)


def estimate_transitions(post, trans=None):
    """  Naive estimate of transition behaviour from posteriors
    :param post: posterior probabilities of kmers by event.
    :param trans: prior belief of transition behaviour (None = use global estimate)
    """
    assert trans is None or len(trans) == 3, 'Incorrect number of transitions'
    res = np.zeros((len(post), 3))
    res[:] = _ETA

    for ev in range(1, len(post)):
        stay = np.sum(post[ev - 1] * post[ev])
        p = post[ev].reshape((-1, _NSTEP))
        step = np.sum(post[ev - 1] * np.tile(np.sum(p, axis=1), _NSTEP)) / _NSTEP
        p = post[ev].reshape((-1, _NSKIP))
        skip = np.sum(post[ev - 1] * np.tile(np.sum(p, axis=1), _NSKIP)) / _NSKIP
        res[ev - 1] = [stay, step, skip]

    if trans is None:
        trans = np.sum(res, axis=0)
        trans /= np.sum(trans)

    res *= trans
    res /= np.sum(res, axis=1).reshape((-1, 1))

    return res
