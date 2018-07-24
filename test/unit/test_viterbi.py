import unittest
import numpy as np
from sloika import viterbi_helpers


class ViterbiTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(0xdeadbeef)
        self.n = 10
        self.slip = 5.0

    def test_001_cython_same_as_python(self):
        x = np.random.normal(size=self.n).astype(np.float32)
        y1s, y1i = viterbi_helpers.slip_update(x, self.slip)
        y2s = np.zeros(len(x), dtype=np.float32)
        y2i = np.zeros(len(x), dtype=np.int)

        y2s[0] = y2s[1] = -1e38
        y2s[2] = x[0] - self.slip
        y2i[2] = 0
        for j in range(3, len(x)):
            if y2s[j - 1] >= x[j - 2]:
                y2s[j] = y2s[j - 1]
                y2i[j] = y2i[j - 1]
            else:
                y2s[j] = x[j - 2]
                y2i[j] = j - 2
            y2s[j] -= self.slip

        np.testing.assert_almost_equal(y1s, y2s)
        np.testing.assert_equal(y1i, y2i)
