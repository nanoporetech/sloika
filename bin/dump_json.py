#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pickle
import sys
import warnings

from sloika.cmdargs import AutoBool, FileExists, FileAbsent

json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--out_file', default=None, action=FileAbsent, help='Output JSON file to this file location')
parser.add_argument('--params', default=True, action=AutoBool, help='Output parameters as well as model structure')

parser.add_argument('model', action=FileExists, help='Model file to read from')

#
# Some numpy types are not serializable to JSON out-of-the-box in Python3 -- need coersion. See
# http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
#


class CustomEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(CustomEncoder, self).default(obj)

if __name__ == "__main__":
    args = parser.parse_args()
    try:
        with open(args.model, 'rb') as fh:
            model = pickle.load(fh)
    except UnicodeDecodeError:
        with open(args.model, 'rb') as fh:
            model = pickle.load(fh, encoding='latin1')
            warnings.warn("Support for python 2 pickles will be dropped: {}".format(args.model))

    json_out = model.json(args.params)

    if args.out_file is not None:
        with open(args.out_file, 'w') as f:
            print("Writing to file: ", args.out_file)
            json.dump(json_out, f, indent=4, cls=CustomEncoder)
    else:
        print(json.dumps(json_out, indent=4, cls=CustomEncoder))
