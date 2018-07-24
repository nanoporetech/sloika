#!/usr/bin/env python3
import argparse
import sys

from sloika.cmdargs import (AutoBool, Bounded, ByteString, FileExists, Maybe,
                               NonNegative, Positive, proportion)

import sloika.tools.chunkify_raw
from sloika.tools.chunkify_raw import raw_chunkify_with_identity_main, raw_chunkify_with_remap_main
from sloika.tools.chunkify_with_identity import chunkify_with_identity_main
from sloika.tools.chunkify_with_remap import chunkify_with_remap_main
from sloika import batch


program_description = "Prepare data for model training and save to hdf5 file"
parser = argparse.ArgumentParser(description=program_description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--alphabet', default=b"ACGT", action=ByteString,
                           help='Alphabet of the sequences')
common_parser.add_argument('--input_strand_list', default=None, action=FileExists,
                           help='Strand summary file containing subset')
common_parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                           help='Number of threads to use when processing data')
common_parser.add_argument('--kmer_len', default=5, metavar='k', type=Positive(int),
                           help='Length of kmer to estimate')
common_parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                           help='Limit number of reads to process')
common_parser.add_argument('--overwrite', default=False, action=AutoBool,
                           help='Whether to overwrite any output files')
common_parser.add_argument('input_folder', action=FileExists,
                           help='Directory containing single-read fast5 files')
common_parser.add_argument('output', help='Output HDF5 file')


common_raw_parser = argparse.ArgumentParser(add_help=False)
common_raw_parser.add_argument('--blanks_percentile', metavar='percentage', default=95,
                               type=Bounded(float, 0, 100),
                               help='Percentile above which to filter out chunks with too many blanks')
common_raw_parser.add_argument('--chunk_len', default=2000, metavar='samples', type=Positive(int),
                               help='Length of each read chunk')
common_raw_parser.add_argument('--normalisation', default=sloika.tools.chunkify_raw.DEFAULT_NORMALISATION,
                               choices=sloika.tools.chunkify_raw.AVAILABLE_NORMALISATIONS,
                               help='Whether to perform median-mad normalisation and with what scope')
common_raw_parser.add_argument('--trim', default=(200, 50), nargs=2, type=NonNegative(int),
                               metavar=('beginning', 'end'),
                               help='Number of samples to trim off start and end')
common_raw_parser.add_argument('--min_length', default=2500, metavar='samples',
                               type=Positive(int), help='Minimum samples in acceptable read')
common_raw_parser.add_argument('--downsample_factor', default=1, type=Positive(int),
                               help='Rate of label downsampling')
common_raw_parser.add_argument('--interpolation', default=False, action=AutoBool,
                               help='Interpolate reference sequence positions between mapped samples')


common_events_parser = argparse.ArgumentParser(add_help=False)
common_events_parser.add_argument('--blanks', metavar='proportion', default=0.7,
                                  type=proportion, help='Maximum proportion of blanks in labels')
common_events_parser.add_argument('--chunk_len', default=500, metavar='events', type=Positive(int),
                                  help='Length of each read chunk')
common_events_parser.add_argument('--normalisation', default=batch.DEFAULT_NORMALISATION,
                                  choices=batch.AVAILABLE_NORMALISATIONS,
                                  help='Whether to perform studentisation and with what scope')
common_events_parser.add_argument('--min_length', default=1200, metavar='events',
                                  type=Positive(int), help='Minimum events in acceptable read')
common_events_parser.add_argument('--use_scaled', default=False, action=AutoBool,
                                  help='Train from scaled event statistics')
common_events_parser.add_argument('--trim', default=(50, 10), nargs=2, type=NonNegative(int),
                                  metavar=('beginning', 'end'),
                                  help='Number of events to trim off start and end')
common_events_parser.add_argument('--section', default='template',
                                  choices=['template', 'complement'], help='Section to call')


common_remap_parser = argparse.ArgumentParser(add_help=False)
common_remap_parser.add_argument('--compile', default=None, type=Maybe(str),
                                 help='File output compiled model')
common_remap_parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
                                 type=proportion, help='Minimum allowed probabiility for basecalls')
common_remap_parser.add_argument('--output_strand_list', default="strand_output_list.txt",
                                 help='Strand summary output file')
common_remap_parser.add_argument('--prior', nargs=2, metavar=('start', 'end'), default=(25.0, 25.0),
                                 type=Maybe(NonNegative(float)), help='Mean of start and end positions')
common_remap_parser.add_argument('--slip', default=5.0, type=Maybe(NonNegative(float)),
                                 help='Slip penalty')
common_remap_parser.add_argument('model', action=FileExists, help='Pickled model file')
common_remap_parser.add_argument('references', action=FileExists,
                                 help='Reference sequences in fasta format')


# add subparsers for each command
subparsers = parser.add_subparsers(help='command', dest='command')
subparsers.required = True


parser_identity = subparsers.add_parser('identity', parents=[common_parser, common_events_parser],
                                        help='Create HDF file from reads as is',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_identity.set_defaults(command_action=chunkify_with_identity_main)


parser_remap = subparsers.add_parser('remap', parents=[common_parser, common_events_parser, common_remap_parser],
                                     help='Create HDF file remapping reads on the fly using transducer network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_remap.add_argument('--segmentation', default='Segmentation',
                          metavar='location', help='Location of segmentation information')
parser_remap.set_defaults(command_action=chunkify_with_remap_main)


parser_raw_identity = subparsers.add_parser('raw_identity', parents=[common_parser, common_raw_parser],
                                            help='Create HDF file from reads as is using raw data',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_raw_identity.set_defaults(command_action=raw_chunkify_with_identity_main)


parser_raw_remap = subparsers.add_parser('raw_remap',
                                         parents=[common_parser, common_raw_parser, common_remap_parser],
                                         help='Create HDF file of raw data, remapping reads on the fly',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_raw_remap.add_argument('--open_pore_fraction', metavar='proportion', default=0.0,
                              type=proportion, help='Max fraction of signal to trim due to open pore')
parser_raw_remap.set_defaults(command_action=raw_chunkify_with_remap_main)


def main(argv):
    args = parser.parse_args(argv[1:])

    try:
        return args.command_action(args)
    except Exception as e:
        print('Exception when running command {!r}.\n{!r}\n'.format(args.command, e))
        raise


if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))
