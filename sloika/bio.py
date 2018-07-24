""" Module containing collection of functions for operating on sequences
represented as strings, and lists thereof.
"""
from sloika.iterators import product, window

# Base complements
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
               '-': '-'}


def all_kmers(length, alphabet='ACGT'):
    """ Find all possible kmers of given length.

    :param length: length of kmers required
    :param alphabet: string from which to draw characters

    :returns: a list of strings. kmers are sorted by the ordering of the *alphabet*.
    """
    if isinstance(alphabet, bytes):
        alphabet = alphabet.decode('utf-8')
        return [''.join(x).encode('utf-8') for x in product(alphabet, repeat=length)]
    else:
        return [''.join(x) for x in product(alphabet, repeat=length)]


def kmer_mapping(length, alphabet='ACGT'):
    """ Dictionary mapping kmer to lexographical order

    :param length: length of kmers required
    :param alphabet: string from which to draw characters

    :returns: a dictionary mapping kmers to lexiographal order, as defined by
    the ordering letters in the *alphabet* argument.
    """
    return {k : i for i, k in enumerate(all_kmers(length, alphabet))}


def all_multimers(length, alphabet='ACGT'):
    """  All possible multimers up to given length

    :param length: maximum length of mulitmer
    :param alphabet: string from which to draw characters

    :returns: a list of strings.  multimers sorted by length then ordering of
    the *alphabet*
    """
    multimers = ['']
    for k in range(length):
        kmers = all_kmers(k + 1, alphabet)
        multimers += kmers
    return multimers


def multimer_mapping(length, alphabet='ACGT'):
    """  All possible multimers up to given length

    :param length: maximum length of mulitmer
    :param alphabet: string from which to draw characters

    :returns: a dictionary mapping multimers to an ordering
    """
    return {k : i for i, k in enumerate(all_multimers(length, alphabet))}


def de_bruijn(k, n, pad=False):
    """ De Bruijn sequence for alphabet size k and subsequences of length n.

    :param k: number of unique symbols
    :param length: length of subsequences
    :param pad: pad end of sequence so cyclic wrap is not required

    .. Note::
       The output must be cyclically wrapped to obtain all unique
       subsequences
    """
    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                for j in range(1, p + 1):
                    sequence.append(a[j])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    if pad:
        sequence += [0] * (n - 1)
    return sequence


def is_homopolymer(k):
    """ Is kmer a homopolymer?

    :param k: A kmer

    :returns: Boolean
    """
    b0 = k[0]
    return all(b == b0 for b in k)


def complement(base, compdict=_COMPLEMENT):
    """ Complement a base

    :param base: A base
    :param compdict: A dictionary containing base complements

    :returns: A base
    """
    return compdict[base]


def reverse_complement(seq, compdict=_COMPLEMENT):
    """ Return reverse complement of a base sequence.

    :param seq: A string of bases.
    :param compdict: A dictionary containing base complements

    :returns: A string of bases.

    """
    return ''.join(compdict[b] for b in seq)[::-1]


def reverse_complement_kmers(kmers, compdict=_COMPLEMENT):
    """ Return reverse complement of kmer sequence, i.e. convert a template
    kmer list to complement equivalent.

    The input list is reversed and each kmer is reversed complemented.

    :param kmers: A list of kmers
    :param compdict: A dictionary containing base complements

    :returns: A list of kmers
    """
    return [reverse_complement(k, compdict) for k in kmers][::-1]


def seq_to_kmers(seq, length):
    """ Turn a string into a list of (overlapping) kmers.

    e.g. perform the transformation:

    'ATATGCG' => ['ATA','TAT', 'ATG', 'TGC', 'GCG']

    :param seq: character string
    :param length: length of kmers in output

    :returns: A list of overlapping kmers
    """
    return [seq[x:x + length] for x in range(0, len(seq) - length + 1)]


def max_overlap(kmers, allow_identical=True):
    """  Determine the maximum overlap from from one kmer to the next

    :param kmers: A iterable of kmers.
    :param allow_identical: Maximum overlap may be entire kmer

    :returns: A list of moves
    """
    res = []
    for k1, k2 in window(kmers, 2):
        move = len(k1)
        if allow_identical and k1 == k2:
            move = 0
        else:
            for i in range(1, len(k1)):
                if k1[i:] == k2[:-i]:
                    move = i
                    break
        res.append(move)
    return res


def moves_compatible(kmers, moves):
    """  Determine whether moves are compatible with list of kmers

    :param kmers: A iterable of kmers.
    :param moves: A iterable of moves from one kmer to the next.

    :returns: A list of booleans
    """

    """
    Deal with three cases
        i. Complete overlap
        ii. Overlap
        iii. No overlap
        The third case ends up being equivalent to the second because of
        Python's string indexing as k1[m:] and k2[:-m] both evaluate to the
        empty string.
    """
    res = []
    for (k1, k2), m in zip(window(kmers, 2), moves):
        res.append((m == 0 and k1 == k2) or (k1[m:] == k2[:-m]))
    return res


def reduce_kmers(kmers, moves):
    """ Reduce a set of kmers to sequence given a set of moves

    :param kmers: a list of kmers
    :param moves: a list of moves

    :returns: a sequence
    """
    assert(all(moves_compatible(kmers, moves))), 'Moves not consistent with kmers'
    kiter = iter(kmers)

    seq = next(kiter)
    for k, m in zip(kiter, moves):
        if m == 0:
            continue
        if m >= len(k):
            seq += k
            continue
        seq += k[-m:]
    return seq


def kmers_to_sequence(kmers, always_move=False):
    """ Produce a sequence from kmer s by maximum overlap

    :param kmers: a list of kmers
    :param always_move: Maximum overlap cannot be entire kmer (stay)

    :returns: a sequence
    """
    moves = max_overlap(kmers, not always_move)
    return reduce_kmers(kmers, moves)


def kmer_transitions(kmers, proposed_max_move, alphabet='ACGT', forward_only=True):
    """Calculate all possible destination kmers from a list of source kmers.

    :param kmers: list of kmers (of equal length)
    :param bases: string specifying alphabet
    :param proposed_max_move: int, maximum number of bases to move
    :param forward_only: bool, if True include kmers ahead of each source kmer

    :returns: {source_kmer: [(move, destination_kmer)]}
    """
    k = len(kmers[0])
    assert all(len(x) == k for x in kmers)
    max_move = min(proposed_max_move, k - 1)

    nmers = [all_kmers(n, alphabet=alphabet) for n in range(max_move + 1)]

    trans = {kmer: list() for kmer in kmers}

    for kmer, move in product(kmers, range(max_move + 1)):
        trans[kmer].extend([(move, kmer[move:] + suffix) for suffix in nmers[move]])
        if not forward_only and move > 0:  # don't want stay twice
            trans[kmer].extend([(-1 * move, suffix + kmer[:-move]) for suffix in nmers[move]])

    return trans
