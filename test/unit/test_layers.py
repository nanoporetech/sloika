import abc
import pickle
import json
import numpy as np
import tempfile
import unittest

import theano as th
import theano.tensor as T

from sloika import activation
from sloika.config import sloika_dtype
import sloika.layers as nn


def rvs(dim):
    '''
    Draw random samples from SO(N)

    Taken from

    http://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
    '''
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


class ANNTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(0xdeadbeef)
        self._NSTEP = 25
        self._NFEATURES = 3
        self._SIZE = 64
        self._NBATCH = 2

        self.W = np.random.normal(size=(self._SIZE, self._NFEATURES)).astype(sloika_dtype)
        self.b = np.random.normal(size=self._SIZE).astype(sloika_dtype)
        self.x = np.random.normal(size=(self._NSTEP, self._NBATCH, self._NFEATURES)).astype(sloika_dtype)
        self.res = self.x.dot(self.W.transpose()) + self.b

    def test_000_single_layer_linear(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.linear)
        network.set_params({'W': self.W, 'b': self.b})
        f = network.compile()
        np.testing.assert_almost_equal(f(self.x), self.res, decimal=5)

    def test_001_single_layer_tanh(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        network.set_params({'W': self.W, 'b': self.b})
        f = network.compile()
        np.testing.assert_almost_equal(f(self.x), np.tanh(self.res), decimal=5)

    def test_002_parallel_layers(self):
        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        l1.set_params({'W': self.W, 'b': self.b})
        l2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        l2.set_params({'W': self.W, 'b': self.b})
        network = nn.Parallel([l1, l2])
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:, :, :self._SIZE], res[:, :, self._SIZE:])

    def test_003_simple_serial(self):
        W2 = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        res = self.res.dot(W2.transpose())

        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                            fun=activation.linear)
        l1.set_params({'W': self.W, 'b': self.b})
        l2 = nn.FeedForward(self._SIZE, self._SIZE, fun=activation.linear)
        l2.set_params({'W': W2})
        network = nn.Serial([l1, l2])
        f = network.compile()

        np.testing.assert_almost_equal(f(self.x), res, decimal=4)

    def test_004_reverse(self):
        network1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        network1.set_params({'W': self.W, 'b': self.b})
        f1 = network1.compile()
        res1 = f1(self.x)
        network2 = nn.Reverse(network1)
        f2 = network2.compile()
        res2 = f2(self.x)

        np.testing.assert_almost_equal(res1, res2)

    def test_005_poormans_birnn(self):
        layer1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        layer1.set_params({'W': self.W, 'b': self.b})
        layer2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        layer2.set_params({'W': self.W, 'b': self.b})
        network = nn.birnn(layer1, layer2)
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:, :, :self._SIZE], res[:, :, self._SIZE:])

    def test_006_softmax(self):
        network = nn.Softmax(self._NFEATURES, self._SIZE, has_bias=True)
        network.set_params({'W': self.W, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        res_sum = res.sum(axis=2)
        self.assertTrue(np.allclose(res_sum, 1.0))

    def test_007_rnn_no_state(self):
        sW = np.zeros((self._SIZE, self._SIZE), dtype=sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                               fun=activation.linear)
        network.set_params({'iW': self.W, 'sW': sW, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res, self.res, decimal=5)

    def test_008_rnn_no_input(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE)
        network.set_params({'iW': iW, 'sW': sW})
        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res, 0.0)

    def test_009_rnn_no_input_with_bias(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = rvs(self._SIZE).astype(sloika_dtype)
        network = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                               fun=activation.linear)
        network.set_params({'iW': iW, 'sW': sW, 'b': self.b})
        f = network.compile()

        res = f(self.x)
        res2 = np.zeros((self._NBATCH, self._SIZE), dtype=sloika_dtype)
        for i in range(self._NSTEP):
            res2 = res2.dot(sW.transpose()) + self.b
            np.testing.assert_almost_equal(res[i], res2)

    def test_010_birnn_no_input_with_bias(self):
        iW = np.zeros((self._SIZE, self._NFEATURES), dtype=sloika_dtype)
        sW = np.random.normal(size=(self._SIZE, self._SIZE)).astype(sloika_dtype)
        layer1 = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                              fun=activation.linear)
        layer1.set_params({'iW': iW, 'sW': sW, 'b': self.b})
        layer2 = nn.Recurrent(self._NFEATURES, self._SIZE, has_bias=True,
                              fun=activation.linear)
        layer2.set_params({'iW': iW, 'sW': sW, 'b': self.b})
        network = nn.birnn(layer1, layer2)

        f = network.compile()

        res = f(self.x)
        np.testing.assert_almost_equal(res[:, :, :self._SIZE], res[::-1, :, self._SIZE:])

    def test_012_simple_derivative(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE,
                                 fun=activation.linear)
        network.set_params({'W': self.W})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, params)
        f = th.function([x], grad)

        theano_grad = f(self.x)[0]
        analytic_grad = np.sum(self.x, axis=(0, 1))
        self.assertEqual(theano_grad.shape, (self._SIZE, self._NFEATURES))
        for i in range(self._NFEATURES):
            np.testing.assert_almost_equal(theano_grad[i], analytic_grad, decimal=3)

    def test_013_simple_derivative_with_bias(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.linear)
        network.set_params({'W': self.W, 'b': self.b})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, params)
        f = th.function([x], grad)

        theano_grad = f(self.x)
        W_grad = np.sum(self.x, axis=(0, 1))
        self.assertEqual(theano_grad[0].shape, (self._SIZE, self._NFEATURES))
        self.assertEqual(theano_grad[1].shape[0], self._SIZE)
        for i in range(self._NFEATURES):
            np.testing.assert_almost_equal(theano_grad[0][i], W_grad, decimal=3)
            np.testing.assert_almost_equal(theano_grad[1][i], self._NBATCH * self._NSTEP)

    def test_014_complex_derivative(self):
        iW = np.random.normal(size=(4, self._SIZE, self._NFEATURES)).astype(sloika_dtype)
        sW = np.random.normal(size=(4, self._SIZE, self._SIZE)).astype(sloika_dtype)
        b = np.random.normal(size=(4, self._SIZE)).astype(sloika_dtype)
        p = np.random.normal(size=(3, self._SIZE)).astype(sloika_dtype)
        network = nn.Lstm(self._NFEATURES, self._SIZE, has_bias=True, has_peep=True)
        network.set_params({'iW': iW, 'sW': sW, 'b': b, 'p': p})

        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.gradient.jacobian(loss, network.params())
        f = th.function([x], grad)

        theano_grad = f(self.x)[0]

    def test_015_save_then_load(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE,
                                 fun=activation.linear)
        network.set_params({'W': self.W})
        params = network.params()
        x = T.tensor3()
        loss = T.sum(network.run(x))
        grad = th.grad(loss, params)
        f = th.function([x], grad)
        with tempfile.TemporaryFile() as tf:
            pickle.dump(f, tf)
            tf.seek(0)
            f2 = pickle.load(tf)

        theano_grad = f(self.x)[0]
        theano_grad2 = f2(self.x)[0]
        self.assertEqual(theano_grad.shape, (self._SIZE, self._NFEATURES))
        self.assertEqual(theano_grad2.shape, (self._SIZE, self._NFEATURES))
        np.testing.assert_almost_equal(theano_grad, theano_grad2, decimal=3)

    def test_016_window(self):
        _WINLEN = 3
        network = nn.Window(self._NFEATURES, _WINLEN)
        f = network.compile()
        res = f(self.x)
        #  Window is now 'SAME' not 'VALID'. Trim
        wh = _WINLEN // 2
        res = res[wh: -wh]
        for j in range(self._NBATCH):
            for i in range(_WINLEN - 1):
                try:
                    np.testing.assert_almost_equal(
                        res[:, j, i * _WINLEN: (i + 1) * _WINLEN], self.x[i: 1 + i - _WINLEN, j])
                except:
                    win_max = np.amax(np.fabs(res[:, :, i * _WINLEN: (i + 1) * _WINLEN] - self.x[i: 1 + i - _WINLEN]))
                    print("Window max: {}".format(win_max))
                    raise
            np.testing.assert_almost_equal(res[:, j, _WINLEN * (_WINLEN - 1):], self.x[_WINLEN - 1:, j])
            #  Test first and last rows explicitly
            np.testing.assert_almost_equal(self.x[:_WINLEN, j].ravel(), res[0, j].transpose().ravel())
            np.testing.assert_almost_equal(self.x[-_WINLEN:, j].ravel(), res[-1, j].transpose().ravel())

    @unittest.skip('Decoding needs fixing')
    def test_017_decode_simple(self):
        _KMERLEN = 3
        network = nn.Decode(_KMERLEN)
        f = network.compile()
        res = f(self.res)

    def test_018_studentise(self):
        network = nn.Studentise(self._NFEATURES)
        f = network.compile()
        res = f(self.x)

        np.testing.assert_almost_equal(np.mean(res, axis=(0, 1)), 0.0)
        np.testing.assert_almost_equal(np.std(res, axis=(0, 1)), 1.0, decimal=4)

    def test_019_identity(self):
        network = nn.Identity(self._NFEATURES)
        f = network.compile()
        res = f(self.res)

        np.testing.assert_almost_equal(res, self.res)


class LayerTest(metaclass=abc.ABCMeta):
    """Mixin abstract class for testing basic layer functionality
    Writing a TestCase for a new layer is easy, for example:

    class RecurrentTest(LayerTest, unittest.TestCase):
        # Inputs for testing the Layer.run() method
        _INPUTS = [np.zeros((10, 20, 12)),
                   np.random.uniform(size=(10, 20, 12)),]
        # Names of the learned parameters of the layer
        _PARAMS = ['iW', 'sW',]

        # The setUp method should instantiate the layer
        def setUp(self):
            self.layer = nn.Recurrent(12, 64)
    """

    _INPUTS = None  # List of input matrices for testing the layer's run method
    _PARAMS = None  # List of names for the learned parameters of the layer

    @abc.abstractmethod
    def setUp(self):
        """Create the layer as self.layer"""
        return

    def test_000_run(self):
        if self._INPUTS is None:
            raise NotImplementedError("Please specify layer inputs for testing, or explicitly skip this test.")
        f = self.layer.compile()
        outs = [f(In.astype(sloika_dtype)) for In in self._INPUTS]

    def test_002_json_dumps(self):
        js = json.dumps(self.layer.json())
        js2 = json.dumps(self.layer.json(params=True))

    def test_003_json_decodes(self):
        props = json.JSONDecoder().decode(json.dumps(self.layer.json()))
        props2 = json.JSONDecoder().decode(json.dumps(self.layer.json(params=True)))

    def test_004_get_set_params(self):
        if self._PARAMS is None:
            raise NotImplementedError(
                "Please specify names of layer parameters, or explicitly set to [] if there are none.")
        p0 = self.layer.json(params=True)["params"]
        p0 = {p: np.array(v, dtype=sloika_dtype) for (p, v) in list(p0.items())}
        for (p, v) in list(p0.items()):
            p0[p] = v + np.random.uniform(size=v.shape).astype(sloika_dtype)
        self.layer.set_params(p0)
        p1 = self.layer.json(params=True)["params"]
        p1 = {p: np.array(v) for (p, v) in list(p1.items())}
        self.assertTrue(all([np.allclose(p0[k], p1[k]) for k in self._PARAMS]))

    def test_005_should_have_params_method(self):
        plist = self.layer.params()

    def test_006_should_have_insize_property(self):
        insize = self.layer.insize

    def test_007_should_have_size_property(self):
        size = self.layer.size

    def test_008_should_have_name_property(self):
        name = self.layer.name


class RecurrentTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', ]

    def setUp(self):
        self.layer = nn.Recurrent(12, 64)


class RecurrentBiasedTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', 'b', ]

    def setUp(self):
        self.layer = nn.Recurrent(12, 64, has_bias=True)


class LstmTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', ]

    def setUp(self):
        self.layer = nn.Lstm(12, 64)


class LstmCIFGTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', ]

    def setUp(self):
        self.layer = nn.LstmCIFG(12, 64)


class LstmOTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', ]

    def setUp(self):
        self.layer = nn.LstmO(12, 64)


class Mut1Test(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['W_xu', 'W_xz', 'W_xr', 'W_hr', 'W_hh', ]

    def setUp(self):
        self.layer = nn.Mut1(12, 64)


class Mut2Test(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['W_xu', 'W_xz', 'W_hz', 'W_hr', 'W_hh', 'W_xh', ]

    def setUp(self):
        self.layer = nn.Mut2(12, 64)


class Mut3Test(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['W_xu', 'W_xz', 'W_hz', 'W_xr', 'W_hr', 'W_hh', 'W_xh', ]

    def setUp(self):
        self.layer = nn.Mut3(12, 64)


class GruTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['iW', 'sW', 'sW2', ]

    def setUp(self):
        self.layer = nn.Gru(12, 64)


class ScrnTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['isW', 'sfW', 'ifW', 'ffW', ]

    def setUp(self):
        self.layer = nn.Scrn(12, 48, 16)


class GenmutTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]
    _PARAMS = ['xW', 'sW', 'sW2']

    def setUp(self):
        self.layer = nn.Genmut(12, 64)


class ConvolutionTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]
    _PARAMS = ['W', 'b']

    def setUp(self):
        self.layer = nn.Convolution(12, 32, 11, 5, has_bias=True)


class MaxPoolTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]
    _PARAMS = []

    def setUp(self):
        self.layer = nn.MaxPool(12, 5, 5)

    @unittest.skip('No params to get or set')
    def test_004_get_set_params(self):
        pass
