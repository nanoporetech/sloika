DEFAULT_ALPHABET=b'ACGT'
DEFAULT_NBASE=len(DEFAULT_ALPHABET)


def nkmer(kmer, nbase=DEFAULT_NBASE):
    """  Number of possible kmers of a given length

    :param kmer: Length of kmer
    :param nbase: Number of letters in alphabet

    :returns: Number of kmers
    """
    return nbase ** kmer


def nstate(kmer, transducer=True, bad_state=True, nbase=DEFAULT_NBASE):
    """  Number of states in model

    :param kmer: Length of kmer
    :param transducer: Is the model a transducer?
    :param bad_state: Does the model have a bad state
    :param nbase: Number of letters in alphabet

    :returns: Number of states
    """
    return nkmer(kmer, nbase=nbase) + (transducer or bad_state)
