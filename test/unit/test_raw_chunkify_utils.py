import numpy as np
import unittest

from sloika.config import sloika_dtype
from sloika.tools.chunkify_raw import convert_mapping_times_to_samples
from sloika.util import is_close


class RawChunkifyUtilTest(unittest.TestCase):

    def test_convert_mapping_times_to_samples(self):
        events = np.array([(99.80953125, 11355.1985 , 0.00075, 'TTTGCC', 1),
                           (102.37074939, 11355.19925, 0.0025 , 'TTTGCC', 0),
                           (88.89286377, 11355.20175, 0.0015 , 'TTGCCG', 1),
                           (89.49268066, 11355.20325, 0.0015 , 'TTGCCG', 0),
                           (104.62006274, 11355.20475, 0.00125, 'TGCCGA', 1)],
                          dtype=[('mean', '<f8'), ('start', '<f8'), ('length', '<f8'), ('kmer', 'S6'), ('move', '<i8')])
        raw = np.array([96.99039185, 98.42995239, 97.71017212, 97.53022705,
                        97.35028198, 95.73077637, 95.5508313 , 99.14973267,
                        98.9697876 , 101.30907349, 104.72802979, 104.72802979,
                        102.56868896, 106.52748047, 103.10852417, 101.84890869,
                        103.10852417, 100.94918335, 95.37088623, 100.76923828,
                        90.87225952, 89.97253418, 87.45330322, 88.53297363,
                        88.89286377, 87.63324829, 90.51236938, 89.97253418,
                        87.99313843, 90.51236938, 87.45330322, 90.51236938,
                        101.48901855, 105.08791992, 107.06731567, 107.78709595,
                        101.66896362, 96.27061157, 96.27061157, 95.37088623,
                        90.87225952, 79.35577515, 86.01374268, 94.11127075,
                        71.97802734, 71.25824707, 74.31731323, 71.79808228,
                        73.9574231])
        sample_rate = 4000.0
        start_sample = 45420787
        commensurate_events = convert_mapping_times_to_samples(events, start_sample, sample_rate)

        self.assertTrue(commensurate_events.dtype == [
                        ('mean', '<f8'), ('start', '<i8'), ('length', '<i8'), ('kmer', 'S6'), ('move', '<i8')])
        self.assertTrue(len(events) == len(commensurate_events))
        for e in commensurate_events:
            self.assertTrue(is_close(raw[e['start']: e['start'] + e['length']].mean(), e['mean']))
