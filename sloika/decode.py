import numpy as np
import sloika.variables as sv


def argmax(post, zero_is_blank=True):
    """  Argmax decoding of simple transducer

    :param post: A 2D :class:`ndarray`
    :param zero_is_blank: Zero is blank state

    :returns: A 1D :class:`ndarray` containing called sequence
    """
    blank_state = 0 if zero_is_blank else post.shape[1] - 1
    path = np.argmax(post, axis=1)
    path_trimmed = path[path != blank_state]
    if zero_is_blank:
        path_trimmed -= 1
    return path_trimmed


def prepare_post(post, min_prob=1e-5, drop_bad=False):
    """ Sanitised posterior matrix for decoding

    :param post: posterior matrix
    :param min_prob: very small posterior probabilies might cause numerical
        problems, so post is replaced by (min_prob + (1 - min_prob) * post)
    :param drop_bad: if True, positions where the most likely state is 0
        are dropped. The 0 state is removed and post is renormalised
    """
    post = np.squeeze(post, axis=1)
    if drop_bad:
        maxcall = np.argmax(post, axis=1)
        post = post[maxcall > 0, 1:]
        weight = np.sum(post, axis=1, keepdims=True)
        post /= weight
    return min_prob + (1.0 - min_prob) * post


def viterbi(post, klen, skip_pen=0.0, log=False, nbase=4):
    """  Viterbi decoding of a kmer transducer

    :param post: A 2d :class:`ndarray`
    :param klen: Length of kmer
    :param log: post array is in log space

    :returns:
    """
    _ETA = 1e-10
    nev, nst = post.shape
    assert klen >= 3, "Kmer not long enough to apply Viterbi with skips"
    nkmer = sv.nkmer(klen, nbase=nbase)
    assert sv.nstate(klen, transducer=True, nbase=nbase) == nst
    nstep = nbase
    nskip = nbase ** 2

    lpost = np.log(post + _ETA) if not log else post
    vscore = lpost[0][1:].copy()
    pscore = np.empty(nkmer)
    traceback = np.empty((nev, nkmer), dtype=np.int16)
    for i in range(1, nev):
        #  Forwards Viterbi iteration
        pscore, vscore = vscore, pscore

        #  Step
        pscore = pscore.reshape(nstep, -1)
        nrem = pscore.shape[1]
        score_step = np.repeat(np.amax(pscore, axis=0), nstep)
        from_step = np.repeat(nrem * np.argmax(pscore, axis=0) + list(range(nrem)), nstep)
        #  Skip
        pscore = pscore.reshape(nskip, -1)
        nrem = pscore.shape[1]
        score_skip = np.repeat(np.amax(pscore, axis=0), nskip) - skip_pen
        from_skip = np.repeat(nrem * np.argmax(pscore, axis=0) + list(range(nrem)), nskip)
        #  Best score for step and skip
        vscore = lpost[i][1:] + np.maximum(score_step, score_skip)
        traceback[i] = np.where(score_step > score_skip, from_step, from_skip)

        #  Stay -- set traceback to be negative
        pscore = pscore.reshape(-1)
        score_stay = pscore + lpost[i][0]
        traceback[i] = np.where(vscore > score_stay, traceback[i], -1)
        vscore = np.maximum(vscore, score_stay)

    stseq = np.empty(nev, dtype=np.int16)
    seq = [np.argmax(vscore)]
    for i in range(nev - 1, 0, -1):
        #  Viterbi traceback
        tstate = traceback[i][seq[-1]]
        if tstate >= 0:
            seq.append(tstate)
        stseq[i - 1] = tstate

    return np.amax(vscore), seq[::-1]


def score(post, seq, full=False):
    """  Compute score of a sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param full: Force full length mapping

    :returns: score
    """
    return forwards(post, seq, full=full)


def forwards(post, seq, full=False):
    """ The forwards score for sequence

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param full: Force full length mapping

    :returns: score
    """
    seq_len = len(seq)

    #  Use seq_len + 1 since additional 'blank state' at beginning
    fwd = np.ones(seq_len + 1)
    fprev = np.ones(seq_len + 1)
    if full:
        fwd .fill(0.0)
        fwd[0] = 1.0
    score = 0.0

    for p in post:
        fwd, fprev = fprev, fwd

        #  Emit blank and stay in current state
        fwd = fprev * p[-1]
        #  Move from previous state and emit new character
        fwd[1:] += fprev[:-1] * p[seq]

        m = np.sum(fwd)
        fwd /= m
        score += np.log(m)

    return score + (np.log(fwd[-1]) if full else 0.0)


def forwards_transpose(post, seq, skip_prob=0.0):
    """ Forwards score but computed through sequence

    Demonstrate that the forward score for a transducer can be computed by
    iterating through the sequence.  This shows the possibility of an efficient
    iterative refinement of the sequence.

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param skip_prob: Probability of skip

    :returns: score
    """
    nev, nstate = post.shape

    fprev = np.zeros(nev + 1)
    fwd = np.concatenate(([1.0], np.cumprod(post[:, -1])))
    m = np.sum(fwd)
    fwd /= m
    score = np.log(m)

    for s in seq:
        fwd, fprev = fprev, fwd

        # Iteration through sequence
        fwd = fprev * skip_prob
        fwd[1:] += fprev[:-1] * post[:, s]
        for i in range(nev):
            fwd[i + 1] += fwd[i] * post[i, -1]

        m = np.sum(fwd)
        fwd /= m
        score += np.log(m)

    return score + np.log(fwd[-1])


def backwards_transpose(post, seq, skip_prob=0.0):
    """ Backwards score computed through sequence

    Demonstrate that the backward score for a transducer can be computed by
    iterating through the sequence.  This shows the possibility of an efficient
    iterative refinement of the sequence.

    :param post: A 2D :class:`ndarray`
    :param seq: Sequence to map against
    :param skip_prob: Probability of skip

    :returns: score
    """
    nev, nstate = post.shape

    bnext = np.zeros(nev + 1)
    bwd = np.concatenate(([1.0], np.cumprod(post[::-1, -1])))[::-1]
    m = np.sum(bwd)
    bwd /= m
    score = np.log(m)

    for s in seq[::-1]:
        bwd, bnext = bnext, bwd

        bwd = bnext * skip_prob
        bwd[:-1] += bnext[1:] * post[:, s]
        for i in range(nev, 0, -1):
            bwd[i - 1] += bwd[i] * post[i - 1, -1]

        m = np.sum(bwd)
        bwd /= m
        score += np.log(m)
    return score + np.log(bwd[0])
