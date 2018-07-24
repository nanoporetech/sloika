#!/usr/bin/env python3
import argparse
import pickle
import sys
from sloika.cmdargs import FileExists
from sloika.layers import Layer
from theano.tensor.sharedvar import TensorSharedVariable
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import theano as th
import warnings

parser = argparse.ArgumentParser(
    'Converts pickled sloika model between CPU and GPU (CUDA) versions.',
    epilog='If you encounter problems please contact joe.harvey@nanoporetech.com')
parser.add_argument('--target', default='swap',
                    choices=('cpu', 'gpu', 'swap'), help='Target device (cpu, gpu or swap)')
parser.add_argument('model', action=FileExists,
                    help='Pickled sloika model to convert')
parser.add_argument('output', help='Output file to write to')


def device(obj, name):
    return ('cpu' if isinstance(getattr(obj, name), TensorSharedVariable)
            else 'gpu')


def get_var_names(obj, depth=0, max_depth=3):
    """Get names and parent objects of all theano shared variables in obj

    Hackety hack hack! Wee!!

    :param obj: object to traverse (depth first) looking for shared variables
    :param depth: depth of recursion so far (internal use only)
    :param max_depth: maximum recursion depth within any Layer instance

    :returns: list of (object, name, device) triples for which object.name is a
        theano shared variable on the specified device
    """
    shared_vars = []
    for name, value in list(vars(obj).items()):
        if isinstance(value, TensorSharedVariable):
            shared_vars.append((obj, name, device(obj, name)))
        elif isinstance(value, CudaNdarraySharedVariable):
            shared_vars.append((obj, name, device(obj, name)))
        elif isinstance(value, Layer):
            shared_vars.extend(get_var_names(value, 0, max_depth=max_depth))
        elif isinstance(value, list):
            # This handles the layers attribute of layers.Serial and
            # layers.Parallel. Max_depth guards against runaway recursion.
            if depth < max_depth:
                for item in value:
                    shared_vars.extend(get_var_names(item, depth + 1,
                                                     max_depth=max_depth))
    return shared_vars

SWAP_WARNING = """
WARNING:
  When --target is 'swap' we expect all shared variables in the model to be
  stored on the same device (cpu or gpu).
  Shared variables were found on multiple devices: {}
  This script will swap each variable between cpu and gpu; if this isn't what
  you want, then set --target explicitly or fix this script ;-)

"""

VAR_CREATE_ERROR = """
ERROR:
  Unable to create shared variable on device '{}'. If targetting the gpu, then
  check your Theano flags and Cuda installation.

"""

if __name__ == '__main__':
    args = parser.parse_args()

    sys.stdout.write('\nLoading pickled model:  {}\n'.format(args.model))
    try:
        with open(args.model, 'rb') as fh:
            net = pickle.load(fh)
    except UnicodeDecodeError:
        with open(args.model, 'rb') as fh:
            net = pickle.load(fh, encoding='latin1')
            warnings.warn("Support for python 2 pickles will be dropped: {}".format(args.model))

    shared_vars = get_var_names(net)

    if len(shared_vars) == 0:
        sys.stderr.write("\nWARNING: Found no shared variables. Nothing to do.\n")
        exit(0)

    objs, names, devices = list(zip(*shared_vars))

    if args.target == 'swap':
        def swap(s):
            return ('cpu' if s == 'gpu' else 'gpu')
        targets = list(map(swap, devices))
        if len(set(targets)) > 1:
            sys.stderr.write(SWAP_WARNING.format(list(set(targets))))
    elif args.target == 'cpu':
        targets = ['cpu' for x in shared_vars]
    elif args.target == 'gpu':
        targets = ['gpu' for x in shared_vars]
    else:
        sys.stderr.write("\nGot unexpected target:  {}\n".format(args.target))
        exit(1)

    for o, n, d, t in zip(objs, names, devices, targets):
        sys.stdout.write(
            "Moving shared variable {}.{} from {} to {}\n".format(o, n, d, t))
        try:
            setattr(o, n, th.shared(getattr(o, n).get_value(), target=t))
        except TypeError:
            sys.stderr.write(VAR_CREATE_ERROR.format(t))
            raise

    sys.stdout.write('\nWriting new pickled model:  {}\n'.format(args.output))
    with open(args.output, 'wb') as fo:
        pickle.dump(net, fo)
