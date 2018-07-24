#!/usr/bin/env python3
import argparse
import pickle
import h5py
import logging
import numpy as np
import sys
import time

import theano as th
import theano.tensor as T

from sloika.cmdargs import (AutoBool, display_version_and_exit, FileExists,
                               Positive)

from sloika.version import __version__

logging.getLogger("theano.gof.compilelock").setLevel(logging.WARNING)

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Validate a simple neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bad', default=True, action=AutoBool,
                    help='Use bad events as a separate state')
parser.add_argument('--batch', default=200, metavar='size', type=Positive(int),
                    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--transducer', default=True, action=AutoBool,
                    help='Model is a transducer')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
                    help='Display version information.')
parser.add_argument('model', action=FileExists,
                    help='File to read model description from')
parser.add_argument('input', action=FileExists,
                    help='HDF5 file containing chunks')


def remove_blanks(labels):
    for lbl_ch in labels:
        for i in range(1, len(lbl_ch)):
            if lbl_ch[i] == 0:
                lbl_ch[i] = lbl_ch[i - 1]
    return labels


def wrap_network(network):
    x = T.tensor3()
    labels = T.imatrix()
    post = network.run(x)
    loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    ncorrect = T.sum(T.eq(T.argmax(post, axis=2), labels))

    fv = th.function([x, labels], [loss, ncorrect])
    return fv


if __name__ == '__main__':
    args = parser.parse_args()

    sys.stdout.write('* Loading network from {}\n'.format(args.model))
    with open(args.model, 'rb') as fh:
        network = pickle.load(fh)
    fv = wrap_network(network)

    sys.stdout.write('* Loading data from {}\n'.format(args.input))
    with h5py.File(args.input, 'r') as h5:
        full_chunks = h5['chunks'][:]
        full_labels = h5['labels'][:]
        full_bad = h5['bad'][:]
    if not args.transducer:
        remove_blanks(full_labels)
    if args.bad:
        full_labels[full_bad] = 0

    total_ev = line_ev = 0
    score = acc = wacc = wscore = 0.0

    t1 = t0 = time.time()
    sys.stdout.write('* Validating\n')
    nbatch = len(full_chunks) // args.batch
    for i in range(nbatch):
        idx = i * args.batch
        events = np.ascontiguousarray(full_chunks[idx : idx + args.batch].transpose((1, 0, 2)))
        labels = np.ascontiguousarray(full_labels[idx : idx + args.batch].transpose())

        fval, ncorr = fv(events, labels)
        fval = float(fval)
        ncorr = float(ncorr)
        nev = np.size(labels)
        line_ev += nev
        total_ev += nev
        score += fval
        wscore += 1
        acc += ncorr
        wacc += nev

        sys.stdout.write('.')

        if (i + 1) % 50 == 0:
            tn = time.time()
            dt = tn - t1
            t = ' {:5d} {:5.3f}  {:5.2f}%  {:5.2f}s ({:.2f} kev/s)\n'
            sys.stdout.write(t.format((i + 1) // 50, score / wscore,
                                      100.0 * acc / wacc, dt, line_ev / 1000.0 / dt))
            line_ev = 0
            t1 = tn

    dt = time.time() - t0
    t = '\nFinal {:5.3f}  {:5.2f}%  {:5.2f}s ({:.2f} kev/s)\n'
    sys.stdout.write(t.format(score / wscore, 100.0 * acc / wacc, dt, total_ev / 1000.0 / dt))
