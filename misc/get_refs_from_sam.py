#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from collections import OrderedDict
import os
import pysam
import sys
import traceback

from sloika.util import fasta_file_to_dict
from sloika.bio import reverse_complement
from sloika.cmdargs import proportion, FileExists


parser = argparse.ArgumentParser(
    description='Extract reference sequence for each read from a SAM alignment file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--coverage', metavar='proportion', default=0.6, type=proportion,
                    help='Minimum coverage')
parser.add_argument('--pad', type=int, default=50,
                    help='Number of bases by which to pad reference sequence')
parser.add_argument('--output_strand_list', default=None,
                    help='Strand list output file')
parser.add_argument('reference', action=FileExists,
                    help="Reference sequence(s) to align against")
parser.add_argument('input', metavar='input.sam',
                    help="SAM file containing read alignments to reference")

STRAND = {0: '+',
          16: '-'}



def trim_fast5_extension(fn):
    """Trim .fast5 extension from filename if present"""
    basename, ext = os.path.splitext(fn)
    return basename if ext == ".fast5" else fn


def get_refs(sam, ref_seq_dict, min_coverage=0.6, pad=50):
    """Read alignments from sam file and return accuracy metrics
    """
    res = []
    with pysam.Samfile(sam, 'r') as sf:
        for read in sf:
            if read.flag != 0 and read.flag != 16:
                continue

            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < min_coverage:
                continue

            start = read.reference_start - read.query_alignment_start - pad
            end = read.reference_end + read.query_length - read.query_alignment_end + pad
            strand = STRAND[read.flag]

            ref = ref_seq_dict.get(sf.references[read.reference_id], None)

            if ref is None:
                continue

            if strand == "+":
                read_ref = str(ref.seq[start:end]).upper()
            else:
                read_ref = reverse_complement(str(ref.seq[start:end]).upper())

            fasta = ">{}\n{}\n".format(trim_fast5_extension(read.qname), read_ref)

            yield (read.qname + ".fast5", fasta)


if __name__ == '__main__':
    args = parser.parse_args()

    sys.stderr.write("* Loading references (this may take a while for large genomes)\n")
    with open(args.reference, 'r') as f:
        seq_records = SeqIO.parse(f, 'fasta')
        references = {r.id: r for r in SeqIO.parse(f, 'fasta')}

    sys.stderr.write("* Extracting read references using SAM alignment\n")
    strand_list = []
    for (name, fasta) in get_refs(args.input, references, args.coverage, args.pad):
        strand_list.append(name)
        sys.stdout.write(fasta)

    if args.output_strand_list is not None:
        sys.stderr.write("* Writing strand-list\n")
        with open(args.output_strand_list, 'w') as f:
            f.write('filename\n')
            f.write('\n'.join(strand_list))
