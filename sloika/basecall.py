import numpy as np
import sys

from fast5_research import Fast5
from sloika import bio
from sloika.maths import mad

from sloika import util
from sloika.variables import nstate, DEFAULT_ALPHABET


def init_worker(model):
    """ Worker init function for basecall_network.py

    This function avoids repeated pickling and unpickling of the model
    by unpickling it once in each process and setting it as a global variable.

    :param model: filename for pickled model to use for basecalling
    """
    import pickle
    global calc_post
    with open(model, 'rb') as fh:
        calc_post = pickle.load(fh)


def decode_post(post, kmer_len, transducer, bad, min_prob, skip=5.0, trans=None, nbase=4, eta=1e-10):
    """ Decode Viterbi state sequence from posterior matrix

    :param post: posterior matrix
    :param kmer_len: kmer length used for decoding
    :param min_prob: passed to prepare_post
    :param transducer: use transducer model
    :param bad: label bad states as 0. If bad and not transducer then time
        points with a bad label will be dropped before decoding
    :param trans: baseline transition probabilities for a non-transducer model
    :param skip: skip penalty for transducer model
    :param eta: small constant for avoiding log(0)
    :param nbase: number of distinct bases

    :returns: score, Viterbi path
    """
    from sloika import decode, olddecode
    assert post.shape[2] == nstate(kmer_len, transducer=transducer, bad_state=bad, nbase=nbase)
    post = decode.prepare_post(post, min_prob=min_prob, drop_bad=bad and not transducer)
    if transducer:
        score, call = decode.viterbi(post, kmer_len, skip_pen=skip, nbase=nbase)
    else:
        assert nbase == 4, "Modified bases not supported by old decoder"
        trans = olddecode.estimate_transitions(post, trans=trans)
        score, call = olddecode.decode_profile(post, trans=np.log(eta + trans), log=False)
    return score, call


def events_worker(fast5_file_name, section, segmentation, trim, kmer_len, transducer,
                  bad, min_prob, alphabet=DEFAULT_ALPHABET, skip=5.0, trans=None):
    """ Worker function for basecall_network.py for basecalling from events

    This worker used the global variable `calc_post` which is set by
    init_worker. `calc_post` is an unpickled compiled sloika model that
    is used to calculate a posterior matrix over states

    :param section: part of read to basecall, 'template' or 'complement'
    :param segmentation: location of segmentation analysis for extracting target read section
    :param trim: (int, int) events to remove from read beginning and end
    :param kmer_len, min_prob, transducer, bad, trans, skip: see `decode_post`
    :param fast5_file_name: filename for single-read fast5 file with event detection and segmentation
    """
    from sloika import features
    try:
        with Fast5(fast5_file_name) as f5:
            ev = f5.get_section_events(section, analysis=segmentation)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting events for section {!r} in file {}\n{!r}\n".format(section, fast5_file_name, e))
        return None

    ev = util.trim_array(ev, *trim)
    if ev.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fast5_file_name))
        return None

    inMat = features.from_events(ev, tag='')[:, None, :]
    score, call = decode_post(calc_post(inMat), kmer_len, transducer, bad, min_prob, skip, trans, nbase=len(alphabet))

    return sn, score, call, inMat.shape[0]


def raw_worker(fast5_file_name, trim, open_pore_fraction, kmer_len, transducer, bad, min_prob,
               alphabet=DEFAULT_ALPHABET, skip=5.0, trans=None):
    """ Worker function for basecall_network.py for basecalling from raw data

    This worker used the global variable `calc_post` which is set by
    init_worker. `calc_post` is an unpickled compiled sloika model that
    is used to calculate a posterior matrix over states

    :param open_pore_fraction: maximum allowed fraction of signal length to
        trim due to classification as open pore signal
    :param trim: (int, int) events to remove from read beginning and end
    :param kmer_len, min_prob, transducer, bad, trans, skip: see `decode_post`
    :param fast5_file_name: filename for single-read fast5 file with raw data
    """
    from sloika import batch, config
    try:
        with Fast5(fast5_file_name) as f5:
            signal = f5.get_read(raw=True)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting raw data for file {}\n{!r}\n".format(fast5_file_name, e))
        return None

    signal = batch.trim_open_pore(signal, open_pore_fraction)
    signal = util.trim_array(signal, *trim)
    if signal.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fast5_file_name))
        return None

    inMat = (signal - np.median(signal)) / mad(signal)
    inMat = inMat[:, None, None].astype(config.sloika_dtype)
    score, call = decode_post(calc_post(inMat), kmer_len, transducer, bad, min_prob, skip, trans, nbase=len(alphabet))

    return sn, score, call, inMat.shape[0]


class SeqPrinter(object):
    """ Formats fasta strings and writes them to stdout or file

    The sequence is calculated on the fly from a Viterbi path of states

    :param kmer_len: length of kmer to use for converting states to kmers
    :param datatype: collective noun for data time used as model input
        e.g. "events" or "samples"
    :param transducer: if True then transitions from a kmer back to itself
        are not allowed when converting kmers to a sequence
    :param fname: name of output file or None to use stdoutto use for converting states to kmers
    :param datatype: collective noun for data time used as model input
        e.g. "events" or "samples"
    :param transducer: if True then transitions from a kmer back to itself
        are not allowed when converting kmers to a sequence
    :param fname: name of output file or None to use sys.stdout
    """
    def __init__(self, kmer_len, datatype="events", transducer=False, fname=None, alphabet=DEFAULT_ALPHABET):
        self.kmers = bio.all_kmers(kmer_len, alphabet=alphabet)
        self.transducer = transducer
        self.datatype = datatype

        if fname is None:
            self.fh = sys.stdout
            self.close_fh = False
        else:
            self.fh = open(fname, 'w')
            self.close_fh = True

    def __del__(self):
        if self.close_fh:
            self.fh.close()

    def write(self, read_name, score, call, nev):
        kmer_path = [self.kmers[i] for i in call]
        seq = bio.kmers_to_sequence(kmer_path, always_move=self.transducer)
        self.fh.write(">{} score {:.0f}, {} {} to {} bases\n".format(read_name, score,
                                                                     nev, self.datatype, len(seq)))
        self.fh.write(seq + '\n')
        return len(seq)
