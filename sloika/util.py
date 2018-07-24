from Bio import SeqIO
import h5py
import numpy as np
import os
import sys


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def geometric_prior(n, m, rev=False):
    """ Log probabilities for random start time with geoemetric distribution

    :param n: length of output vector
    :param m: mean of distribution
    :param rev: reverse distribution

    :returns: A 1D :class:`ndarray` containing log probabilities
    """
    p = 1.0 / (1.0 + m)
    prior = np.repeat(np.log(p), n)
    prior[1:] += np.arange(1, n) * np.log1p(-p)
    if rev:
        prior = prior[::-1]
    return prior


def is_contiguous(ndarray):
    '''
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.ascontiguousarray.html
    '''
    return ndarray.flags['C_CONTIGUOUS']


def get_kwargs(args, names):
    kwargs = {}
    for name in names:
        kwargs[name] = getattr(args, name)
    return kwargs


def progress_report(i):
    """A dotty way of showing progress"""
    i += 1
    sys.stderr.write('.')
    if i % 50 == 0:
        sys.stderr.write('{:8d}\n'.format(i))
    return i


def create_labelled_chunks_hdf5(output, blanks, attributes, chunk_list, label_list, bad_list):
    """ Helper function for chunkify to create hdf5 batch file

    :param chunk_list: event features
    :param label_list: state labels corresponding to chunks in chunk_list
    :param bad_list: bad state masks corresponding to chunks in chunk_list
    """

    assert len(chunk_list) == len(label_list) == len(bad_list)
    assert len(chunk_list) > 0

    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(os.path.normpath(output_dir))

    all_chunks = np.concatenate(chunk_list)
    all_labels = np.concatenate(label_list)
    all_bad = np.concatenate(bad_list)

    #  Mark chunks with too many blanks with a zero weight
    nblank = np.sum(all_labels == 0, axis=1)
    max_blanks = int(all_labels.shape[1] * blanks)
    all_weights = nblank < max_blanks

    with h5py.File(output, 'w') as h5:
        bad_ds = h5.create_dataset('bad', all_bad.shape, dtype='i1',
                                   compression="gzip")
        chunk_ds = h5.create_dataset('chunks', all_chunks.shape, dtype='f4',
                                     compression="gzip")
        label_ds = h5.create_dataset('labels', all_labels.shape, dtype='i4',
                                     compression="gzip")
        weight_ds = h5.create_dataset('weights', all_weights.shape, dtype='f4',
                                      compression="gzip")
        bad_ds[:] = all_bad
        chunk_ds[:] = all_chunks
        label_ds[:] = all_labels
        weight_ds[:] = all_weights

        for (key, value) in attributes.items():
            h5['/'].attrs[key] = value


def trim_array(x, from_start, from_end):
    assert from_start >= 0
    assert from_end >= 0

    from_end = None if from_end == 0 else -from_end
    return x[from_start:from_end]


def fasta_file_to_dict(fasta_file_name):
    """Load records from fasta file as a dictionary"""
    references = dict()
    with open(fasta_file_name, 'r') as fh:
        for ref in SeqIO.parse(fh, 'fasta'):
            refseq = str(ref.seq)
            if 'N' not in refseq and len(refseq) > 0:
                references[ref.id] = refseq.encode('utf-8')

    return references
