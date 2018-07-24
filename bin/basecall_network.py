#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import time

from fast5_research import iterate_fast5
from sloika.cmdargs import (AutoBool, ByteString, FileAbsent, FileExists, Maybe,
                               NonNegative, proportion, Positive, Vector)
from sloika.iterators import imap_mp

from sloika import basecall, helpers, util


# create the top-level parser
parser = argparse.ArgumentParser(
    description='1D basecaller for RNNs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# common command line arguments to all subcommands
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--alphabet', default=b"ACGT", action=ByteString,
                           help='Alphabet of the sequences')
common_parser.add_argument('--compile', default=None, action=FileAbsent,
                           help='File output compiled model')
common_parser.add_argument('--input_strand_list', default=None, action=FileExists,
                           help='Strand summary file containing subset')
common_parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                           help='Number of threads to use when processing data')
common_parser.add_argument('--kmer_len', default=5, metavar='length', type=Positive(int),
                           help='Length of kmer')
common_parser.add_argument('--limit', default=None, metavar='reads',
                           type=Maybe(Positive(int)), help='Limit number of reads to process')
common_parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
                           type=proportion, help='Minimum allowed probabiility for basecalls')
common_parser.add_argument('--skip', default=0.0,
                           type=NonNegative(float), help='Skip penalty')
common_parser.add_argument('--trans', default=None, type=proportion, nargs=3,
                           metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
common_parser.add_argument('--transducer', default=True, action=AutoBool,
                           help='Model is transducer')

common_parser.add_argument('model', action=FileExists, help='Pickled model file')
common_parser.add_argument('input_folder', action=FileExists,
                           help='Directory containing single-read fast5 files')


# add subparsers for each command
subparsers = parser.add_subparsers(help='command', dest='command')
subparsers.required = True

parser_ev = subparsers.add_parser('events', parents=[common_parser], help='basecall from events',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_ev.add_argument('--bad', default=True, action=AutoBool,
                       help='Model emits bad events as a separate state')
parser_ev.add_argument('--section', default='template', choices=['template', 'complement'],
                       help='Section to call')
parser_ev.add_argument('--segmentation', default='Segmentation',
                       metavar='location', help='Location of segmentation information')
parser_ev.add_argument('--trim', default=(50, 1), nargs=2, type=NonNegative(int),
                       metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser_ev.set_defaults(datatype='events')


parser_raw = subparsers.add_parser('raw', parents=[common_parser], help='basecall from raw signal',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_raw.add_argument('--bad', default=True, action=AutoBool,
                        help='Model emits bad signal blocks as a separate state')
parser_raw.add_argument('--open_pore_fraction', metavar='proportion', default=0,
                        type=proportion, help='Max fraction of signal to trim due to open pore')
parser_raw.add_argument('--trim', default=(200, 10), nargs=2, type=NonNegative(int),
                        metavar=('beginning', 'end'), help='Number of samples to trim off start and end')
parser_raw.set_defaults(datatype='samples')


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.datatype in ["events", "samples"]

    assert args.command in ["events", "raw"]

    basecall_worker = getattr(basecall, args.command + "_worker")
    if args.command == "events":
        kwarg_names = ['section', 'segmentation', 'trim', 'kmer_len', 'transducer', 'bad', 'min_prob', 'skip', 'trans', 'alphabet']
    else:
        kwarg_names = ['trim', 'open_pore_fraction', 'kmer_len', 'transducer', 'bad', 'min_prob', 'skip', 'trans', 'alphabet']

    compiled_file = helpers.compile_model(args.model, args.compile)

    seq_printer = basecall.SeqPrinter(args.kmer_len, datatype=args.datatype,
                                      transducer=args.transducer, alphabet=args.alphabet.decode('ascii'))

    files = iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                          strand_list=args.input_strand_list)
    nbases = nevents = 0
    t0 = time.time()
    for res in imap_mp(basecall_worker, files, threads=args.jobs, fix_kwargs=util.get_kwargs(args, kwarg_names),
                       unordered=True, init=basecall.init_worker, initargs=[compiled_file]):
        if res is None:
            continue
        read, score, call, nev = res
        seq_len = seq_printer.write(read, score, call, nev)
        nbases += seq_len
        nevents += nev

    dt = time.time() - t0
    t = 'Called {} bases in {:.1f} s ({:.1f} bases/s or {:.1f} {}/s)\n'
    sys.stderr.write(t.format(nbases, dt, nbases / dt, nevents / dt, args.datatype))

    if compiled_file != args.compile:
        os.remove(compiled_file)
