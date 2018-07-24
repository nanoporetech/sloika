#!/usr/bin/env python3
import argparse
import imp
import logging
import numpy as np
import os
import sys

import theano as th
import theano.tensor as T

from sloika.cmdargs import (display_version_and_exit, FileExists, Positive)

from sloika.version import __version__

logging.getLogger("theano.gof.compilelock").setLevel(logging.WARNING)

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Test compilation and execution of a sloika model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
                    help='Length of kmer')
parser.add_argument('--nfeature', default=4, metavar='number', type=Positive(int),
                    help='Number of features to input to network')
parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
                    help='Standard deviation to initialise with')
parser.add_argument('--stride', default=1, metavar='int', type=Positive(int),
                    help='Stride of model or None for no stride')
parser.add_argument('--winlen', default=3, type=Positive(int),
                    help='Length of window over data')
parser.add_argument('--version', nargs=0, action=display_version_and_exit,
                    metavar=__version__, help='Display version information.')
parser.add_argument('model', action=FileExists,
                    help='Python source file to read model description from')


def wrap_network(network):
    x = T.tensor3()
    labels = T.imatrix()
    post = network.run(x)
    loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    ncorrect = T.sum(T.eq(T.argmax(post, axis=2), labels))

    fg = th.function([x, labels], [loss, ncorrect])
    return fg


if __name__ == '__main__':
    args = parser.parse_args()

    #  Set some Theano options
    th.config.optimizer = 'fast_compile'
    th.config.warn_float64 = 'raise'

    try:
        netmodule = imp.load_source('netmodule', args.model)
        network = netmodule.network(nfeature=args.nfeature, klen=args.kmer, sd=args.sd,
                                    stride=args.stride, winlen=args.winlen)
        fg = wrap_network(network)
    except:
        sys.stderr.write('Compilation of model {} failed\n'.format(args.model))
        raise

    nparam = sum([p.get_value().size for p in network.params()])
    sys.stderr.write('Compilation of model {} succeeded\n'.format(os.path.basename(args.model)))
    sys.stderr.write('nparam = {}\n'.format(nparam))

    for i in range(5):
        ntime = np.random.randint(10, 100)
        nbatch = np.random.randint(2, 10)
        x = np.random.normal(size=(ntime, nbatch, args.nfeature)).astype(th.config.floatX)
        out_length = int(np.ceil(float(ntime) / args.stride))
        lbls = np.zeros((out_length, nbatch), dtype='i4')
        try:
            sys.stderr.write("Input of shape [{}, {}, {}]...  ".format(ntime, nbatch, args.nfeature))
            loss, ncorrect = fg(x, lbls)
            sys.stderr.write("PASS\n")
        except:
            sys.stderr.write('Execution of model {} failed\n'.format(args.model))
            raise
