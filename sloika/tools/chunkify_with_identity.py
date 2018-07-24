import argparse
import os
import sys
import numpy as np

from fast5_research import iterate_fast5
from sloika import batch, util
from sloika.iterators import imap_mp


def chunkify_with_identity_main(args):

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['section', 'chunk_len', 'kmer_len', 'min_length', 'trim', 'use_scaled', 'normalisation']
    i = 0
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(batch.chunk_worker, fast5_files, threads=args.jobs,
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
            'input_type': 'events',
            'kmer': args.kmer_len,
            'normalisation': args.normalisation,
            'scaled': args.use_scaled,
            'section': args.section,
            'trim': args.trim,
            'alphabet': args.alphabet,
        }
        util.create_labelled_chunks_hdf5(args.output, args.blanks, hdf5_attributes, chunk_list, label_list, bad_list)
