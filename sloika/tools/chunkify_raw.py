from Bio import SeqIO
import numpy as np
import os
import sys

from fast5_research import Fast5, iterate_fast5

from sloika import batch, bio, decode, helpers, transducer, util
from sloika.iterators import imap_mp
from sloika.maths import mad


DEFAULT_NORMALISATION = 'per-read'

AVAILABLE_NORMALISATIONS = frozenset(['none', 'per-read', 'per-chunk'])


def convert_mapping_times_to_samples(mapping_table, start_sample, sample_rate):
    """Replace time coordinates in mapping_table with indices into raw signal

    :param mapping_table: array of events (or similar) for a mapped read with
        start times and and lengths measured in seconds
    :param start_sample: start sample of the read raw signal
    :param sample_rate: number of samples per second

    :returns: mapping table with start times measured in samples from the start
        of the raw signal, and lengths measured in samples
    """
    def maybe_change_field_dtype(nd):
        new_field_types = {'start': '<i8', 'length': '<i8'}
        name, dtype = nd
        return (name, new_field_types.get(name, dtype))

    old_dtype = mapping_table.dtype.descr
    new_dtype = list(map(maybe_change_field_dtype, old_dtype))

    assert np.allclose(mapping_table['start'][:-1] + mapping_table['length'][:-1],
                       mapping_table['start'][1:])

    starts = np.around(mapping_table['start'] * sample_rate - start_sample).astype(int)
    lengths = np.around(mapping_table['length'] * sample_rate).astype(int)

    assert np.alltrue(starts[:-1] + lengths[:-1] == starts[1:])

    new_mapping_table = mapping_table.copy().astype(new_dtype)
    new_mapping_table['start'] = starts
    new_mapping_table['length'] = lengths

    return new_mapping_table


def trim_signal_and_mapping(signal, mapping_table, start_sample, end_sample):
    """Trim samples and mapped blocks outside of range [start_sample, end_sample]
    """
    sig_trim = signal[start_sample:end_sample]

    end_sample = start_sample + len(sig_trim)

    ix = np.arange(len(mapping_table))
    lb = int(ix[mapping_table['start'] > start_sample].min()) - 1
    ub = int(ix[mapping_table['start'] < end_sample].max()) + 1
    new_mapping_table = mapping_table[lb:ub].copy()

    new_mapping_table['start'] -= start_sample
    new_mapping_table['start'][0] = 0
    new_mapping_table['length'][0] = new_mapping_table['start'][1]
    new_mapping_table['length'][-1] = len(sig_trim) - new_mapping_table['start'][-1]

    return sig_trim, new_mapping_table


def mapping_table_is_registered(mapped_signal, mapping_table):
    """Test that signal and mapping table cover the same range of samples
    """
    tests = [
        mapping_table['start'][0] == 0,
        mapping_table['start'][-1] + mapping_table['length'][-1] == len(mapped_signal),
        (mapping_table['start'] >= 0).all(),
        (mapping_table['start'] < len(mapped_signal)).all(),
        (mapping_table['start'][:-1] + mapping_table['length'][:-1] == mapping_table['start'][1:]).all(),
    ]
    return all(tests)


def interpolate_pos(mapping_table, att):
    """Return a function: time -> reference position by interpolating mapping

    :param mapping_table: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes direction, ref_start, ref_stop
        (mapping_table, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        EPS = 10**-10  # small value for avoiding round to even

        ev_mid = mapping_table['start'] + 0.5 * mapping_table['length']
        map_k = len(mapping_table['kmer'][0])

        if att['direction'] == "+":
            map_ref_pos = mapping_table['seq_pos'] + 0.5 * map_k - att['ref_start']
        else:
            map_ref_pos = att['ref_stop'] - mapping_table['seq_pos'] + 0.5 * map_k
        pos_interp = np.interp(t, ev_mid, map_ref_pos)
        pos = np.around(pos_interp - 0.5 * k + EPS).astype(np.int)
        return pos

    return interp


def interpolate_labels(mapping_table, att):
    """Return a function: time -> reference kmer by interpolating mapping

    :param mapping_table: mapping table with fields start, length, seq_pos and kmer
    :param att: mapping attributes reference, direction, ref_start, ref_stop
        (mapping_table, att) could be returned by f5file.get_any_mapping_data()
    """
    def interp(t, k=5):
        pos = interpolate_pos(mapping_table, att)(t, k)
        return np.array([batch.kmer_to_state[att['reference'][i: i + k]] for i in pos]) + 1

    return interp


def labels_from_mapping_table(kmer_array, kmer_len, index_from=1):
    """Extract shortened kmers from an array of kmers

    :param kmer_array: a numpy array of kmers
    :param kmer_len: length of sequence context used to determine label

    :returns: an array of labels
    """
    kmer_array = np.ascontiguousarray(kmer_array)

    old_kmer_len = len(kmer_array.flat[0])
    assert kmer_len <= old_kmer_len

    offset = (old_kmer_len - kmer_len + 1) // 2
    extracted = np.chararray(kmer_array.shape, kmer_len, buffer=kmer_array.data,
                             offset=offset, strides=kmer_array.strides)

    labels = np.array(list(map(lambda k: batch.kmer_to_state[k], extracted.flat))) + index_from

    return labels.reshape(kmer_array.shape).astype('i4')


def replace_repeats_with_zero(arr):
    """Replace repeated elements in 1d array with 0"""
    arr[np.ediff1d(arr, to_begin=1) == 0] = 0
    return arr


def fill_zeros_with_prev(arr):
    """Fills non-leading zero values with previous value in 1d array"""
    ix = np.arange(len(arr)) * (arr != 0)
    return arr[np.maximum.accumulate(ix)]


def index_of_previous_non_zero(input_array):
    """output[i] is the index of the last non-zero element in input[:i]"""
    ix = np.arange(len(input_array)) * (input_array > 0)
    output_array = np.maximum.accumulate(ix)
    return output_array


def raw_chunkify(signal, mapping_table, chunk_len, kmer_len, normalisation,
                 downsample_factor, interpolation, mapping_attrs=None):
    """ Generate labelled data chunks from raw signal and mapping table
    """
    assert len(signal) >= chunk_len
    assert normalisation in AVAILABLE_NORMALISATIONS
    assert mapping_table_is_registered(signal, mapping_table)

    ml = len(signal) // chunk_len
    ub = ml * chunk_len
    signal, mapping_table = trim_signal_and_mapping(signal, mapping_table, 0, ub)
    assert mapping_table_is_registered(signal, mapping_table)
    new_inMat = signal.reshape((ml, chunk_len, 1))

    if normalisation == "per-chunk":
        chunk_medians = np.median(new_inMat, axis=1, keepdims=True)
        chunk_mads = mad(new_inMat, axis=1, keepdims=True)
        new_inMat = (new_inMat - chunk_medians) / chunk_mads
    elif normalisation == "per-read":
        new_inMat = (new_inMat - np.median(new_inMat)) / mad(new_inMat)
    else:
        assert normalisation == "none"

    if interpolation:
        block_midpoints = np.arange(0, ub, downsample_factor)
        pos = interpolate_pos(mapping_table, mapping_attrs)(block_midpoints, kmer_len)
        sig_labels = interpolate_labels(mapping_table, mapping_attrs)(block_midpoints, kmer_len)
        sig_labels[np.ediff1d(pos, to_begin=1) == 0] = 0
        sig_labels = sig_labels.reshape((ml, -1))
    else:
        all_labels = labels_from_mapping_table(mapping_table['kmer'], kmer_len)
        labels = all_labels[mapping_table['move'] > 0]
        all_starts = mapping_table['start'][index_of_previous_non_zero(mapping_table['move'])]
        starts = all_starts[mapping_table['move'] > 0]

        idx = np.zeros(ub, dtype=np.int)
        idx[starts] = np.arange(len(labels)) + 1
        idx = fill_zeros_with_prev(idx)
        idx = idx.reshape((ml, chunk_len))[:, ::downsample_factor]
        idx = np.apply_along_axis(replace_repeats_with_zero, 1, idx)

        sig_labels = np.concatenate([[0], labels])[idx].astype('i4')

    # Bad state isn't supported yet with raw models
    sig_bad = np.zeros((ml, chunk_len), dtype=bool)

    return new_inMat, sig_labels, sig_bad


def raw_chunk_worker(fn, chunk_len, kmer_len, min_length, trim, normalisation,
                     downsample_factor, interpolation=False):
    """ Worker for creating labelled features from raw data

    :param fn: A filename to read from.
    :param chunk_len: Length on each chunk
    :param kmer_len: Kmer length for training
    :param min_length: Minumum number of samples before read can be considered.
    :param trim: Tuple (beginning, end) of number of samples to trim from read.
    :param normalisation: Normalisation method [per-chunk | per-read | none]
    :param downsample_factor: factor by which to downsample labels
    :param interpolation: interpolate sequence positions between those in
        mapping table
    """
    try:
        with Fast5(fn) as f5:
            mapping_table, att = f5.get_any_mapping_data('template')
            sig = f5.get_read(raw=True)
            sample_rate = f5.sample_rate
            start_sample = f5.get_read(raw=True, group=True).attrs['start_time']
    except Exception as e:
        sys.stderr.write('Failed to get mapping data from {}.\n{}\n'.format(fn, repr(e)))
        return None

    mapping_table = convert_mapping_times_to_samples(mapping_table, start_sample, sample_rate)
    map_start = mapping_table['start'][0] + trim[0]
    map_end = mapping_table['start'][-1] + mapping_table['length'][-1] - trim[1]
    mapped_signal, mapping_table = trim_signal_and_mapping(sig, mapping_table, map_start, map_end)

    try:
        assert mapping_table_is_registered(mapped_signal, mapping_table)
    except Exception as e:
        sys.stderr.write('Failed to properly register raw signal and mapping table in {}.\n{}\n'.format(fn, repr(e)))
        return None

    if len(mapped_signal) < max(chunk_len, min_length):
        sys.stderr.write('{} is too short.\n'.format(fn))
        return None

    new_inMat, sig_labels, sig_bad = raw_chunkify(mapped_signal, mapping_table, chunk_len, kmer_len, normalisation,
                                                  downsample_factor, interpolation, att)

    return (np.ascontiguousarray(new_inMat),
            np.ascontiguousarray(sig_labels),
            np.ascontiguousarray(sig_bad))


def raw_remap(ref, signal, min_prob, kmer_len, prior, slip):
    """ Map raw signal to reference sequence using transducer model"""
    from sloika import config  # local import to avoid CUDA init in main thread

    inMat = (signal - np.median(signal)) / mad(signal)
    inMat = inMat[:, None, None].astype(config.sloika_dtype)
    post = decode.prepare_post(batch.calc_post(inMat), min_prob=min_prob, drop_bad=False)

    kmers = np.array(bio.seq_to_kmers(ref, kmer_len))
    seq = [batch.kmer_to_state[k] + 1 for k in kmers]
    prior0 = None if prior[0] is None else util.geometric_prior(len(seq), prior[0])
    prior1 = None if prior[1] is None else util.geometric_prior(len(seq), prior[1], rev=True)

    score, path = transducer.map_to_sequence(post, seq, slip=slip, prior_initial=prior0,
                                             prior_final=prior1, log=False)

    mapping_dtype = [
        ('start', '<i8'),
        ('length', '<i8'),
        ('seq_pos', '<i8'),
        ('move', '<i8'),
        ('kmer', 'S{}'.format(kmer_len)),
        ('good_emission', '?'),
    ]
    mapping_table = np.zeros(post.shape[0], dtype=mapping_dtype)
    stride = int(np.ceil(signal.shape[0] / float(post.shape[0])))
    mapping_table['start'] = np.arange(0, signal.shape[0], stride, dtype=np.int) - stride // 2
    mapping_table['length'] = stride
    mapping_table['seq_pos'] = path
    mapping_table['move'] = np.ediff1d(path, to_begin=1)
    mapping_table['kmer'] = kmers[path]
    # We set 'good_emission' for compatability only
    mapping_table['good_emission'] = True

    _, mapping_table = trim_signal_and_mapping(signal, mapping_table, 0, len(signal))

    return (score, mapping_table, path, seq)


def raw_chunk_remap_worker(fn, trim, min_prob, kmer_len, min_length,
                           prior, slip, chunk_len, normalisation, downsample_factor,
                           interpolation, open_pore_fraction, references):
    """ Worker function for `chunkify raw_remap` remapping reads using raw signal"""
    try:
        with Fast5(fn) as f5:
            signal = f5.get_read(raw=True)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write('Failure reading events from {}.\n{}\n'.format(fn, repr(e)))
        return None

    try:
        read_ref = references[sn]
    except Exception as e:
        sys.stderr.write('No reference found for {}.\n{}\n'.format(fn, repr(e)))
        return None

    signal = batch.trim_open_pore(signal, open_pore_fraction)
    signal = util.trim_array(signal, *trim)

    if len(signal) < max(chunk_len, min_length):
        sys.stderr.write('{} is too short.\n'.format(fn))
        return None

    try:
        (score, mapping_table, path, seq) = raw_remap(read_ref, signal, min_prob, kmer_len, prior, slip)
    except Exception as e:
        sys.stderr.write("Failure remapping read {}.\n{}\n".format(sn, repr(e)))
        return None
    # mapping_attrs required if using interpolation
    mapping_attrs = {
        'reference': read_ref,
        'direction': '+',
        'ref_start': 0,
    }
    (chunks, labels, bad_ev) = raw_chunkify(signal, mapping_table, chunk_len, kmer_len, normalisation,
                                            downsample_factor, interpolation, mapping_attrs)

    return sn + '.fast5', score, len(mapping_table), path, seq, chunks, labels, bad_ev


def raw_chunkify_with_identity_main(args):
    """ Main function for `chunkify.py raw_identity` producing batch file for model training
    """
    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['chunk_len', 'kmer_len', 'min_length', 'trim', 'normalisation', 'downsample_factor', 'interpolation']
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(raw_chunk_worker, fast5_files, threads=args.jobs,
                       unordered=True, fix_kwargs=util.get_kwargs(args, kwarg_names),
                       init=batch.init_chunk_identity_worker, initargs=[args.kmer_len, args.alphabet]):
        if res is not None:
            i = util.progress_report(i)

            (chunks, labels, bad_ev) = res

            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)

    if chunk_list == []:
        print("no chunks were produced", file=sys.stderr)
        sys.exit(1)
    else:
        print('\n* Writing out to HDF5')
        hdf5_attributes = {
            'chunk': args.chunk_len,
            'downsample_factor': args.downsample_factor,
            'input_type': 'raw',
            'interpolation': args.interpolation,
            'kmer': args.kmer_len,
            'normalisation': args.normalisation,
            'section': 'template',
            'trim': args.trim,
            'alphabet': args.alphabet,
        }
        blanks_per_chunk = np.concatenate([(l == 0).mean(1) for l in label_list])
        blanks = np.percentile(blanks_per_chunk, args.blanks_percentile)
        util.create_labelled_chunks_hdf5(args.output, blanks, hdf5_attributes, chunk_list, label_list, bad_list)


def raw_chunkify_with_remap_main(args):
    """ Main function for `chunkify.py raw_remap` producing batch file for model training
    """
    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)
        if os.path.exists(args.output_strand_list):
            print("Cowardly refusing to overwrite {}".format(args.output_strand_list))
            sys.exit(2)

    fast5_files = iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.input_strand_list)

    references = util.fasta_file_to_dict(args.references)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['trim', 'min_prob', 'kmer_len', 'min_length',
                   'prior', 'slip', 'chunk_len', 'normalisation', 'downsample_factor',
                   'interpolation', 'open_pore_fraction']
    kwargs = util.get_kwargs(args, kwarg_names)
    kwargs['references'] = references

    i = 0
    compiled_file = helpers.compile_model(args.model, args.compile)
    output_strand_list_entries = []
    bad_list = []
    chunk_list = []
    label_list = []
    with open(args.output_strand_list, 'w') as slfh:
        slfh.write(u'\t'.join(['filename', 'nblocks', 'score', 'nstay', 'seqlen', 'start', 'end']) + u'\n')
        for res in imap_mp(raw_chunk_remap_worker, fast5_files, threads=args.jobs,
                        fix_kwargs=kwargs, unordered=True, init=batch.init_chunk_remap_worker,
                        initargs=[compiled_file, args.kmer_len, args.alphabet]):
            if res is not None:
                i = util.progress_report(i)

                read, score, nblocks, path, seq, chunks, labels, bad_ev = res

                chunk_list.append(chunks)
                label_list.append(labels)
                bad_list.append(bad_ev)
                strand_data = [read, nblocks, -score / nblocks,
                               np.sum(np.ediff1d(path, to_begin=1) == 0),
                               len(seq), min(path), max(path)]
                slfh.write('\t'.join([str(x) for x in strand_data]) + '\n')

    if compiled_file != args.compile:
        os.remove(compiled_file)

    if chunk_list == []:
        print("no chunks were produced", file=sys.stderr)
        sys.exit(1)
    else:
        print('\n* Writing out to HDF5')
        hdf5_attributes = {
            'chunk': args.chunk_len,
            'downsample_factor': args.downsample_factor,
            'input_type': 'raw',
            'interpolation': args.interpolation,
            'kmer': args.kmer_len,
            'normalisation': args.normalisation,
            'section': 'template',
            'trim': args.trim,
            'alphabet': args.alphabet,
        }
        blanks_per_chunk = np.concatenate([(l == 0).mean(1) for l in label_list])
        blanks = np.percentile(blanks_per_chunk, args.blanks_percentile)
        util.create_labelled_chunks_hdf5(args.output, blanks, hdf5_attributes, chunk_list, label_list, bad_list)
