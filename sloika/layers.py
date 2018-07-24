import abc
from collections import OrderedDict
import theano as th
import theano.tensor as T
import numpy as np

from sloika import activation, conv
from sloika.config import sloika_dtype
from sloika.variables import DEFAULT_NBASE, nkmer
from functools import reduce


"""  Convention: inMat row major (C ordering) as (time, batch, state)
"""
_NSTEP = DEFAULT_NBASE
_NSKIP = DEFAULT_NBASE * DEFAULT_NBASE
_FORGET_BIAS = 2.0
_INDENT = ' ' * 4


def zeros(size):
    return np.zeros(size, dtype=sloika_dtype)


def _extract(x, shape=None):
    xv = x.get_value()
    if shape is not None:
        xv = xv.reshape(shape)
    return xv.tolist()


class Layer(metaclass=abc.ABCMeta):

    def compile(self):
        x = T.tensor3()
        return th.function([th.In(x, borrow=True)], th.Out(self.run(x), borrow=True))

    @property
    def insize(self):
        return self._insize

    @property
    def size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def params(self):
        """ a list of network parameters
        """
        return

    @abc.abstractmethod
    def json(self, params):
        """ emit json string describing layer
        """
        return

    @abc.abstractmethod
    def set_params(self, values):
        """ Set parameters from a dictionary of values
        """
        return

    @abc.abstractmethod
    def run(self, inMat):
        """  Run network layer
        """
        return


class RNN(Layer):

    @abc.abstractmethod
    def step(self, in_vec, state):
        """ A single step along the RNN
        :param in_vec: Input to node
        :param state: Hidden state from previous node
        """
        return

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat, outputs_info=T.zeros((nbatch, self.size)))
        return out


class Identity(Layer):

    def __init__(self, insize, name="Identity"):
        self._insize = insize
        self._name = name

    @property
    def size(self):
        return self.insize

    def params(self):
        return []

    def json(self):
        return {'type' : "identity"}

    def set_params(self, values):
        return

    def run(self, inMat):
        return inMat


class FeedForward(Layer):
    """  Basic feedforward layer
         out = f( inMat W + b )

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, name="Feed-forward"):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(size + insize))
        self._insize = insize
        self._size = size
        self._name = name
        self.fun = fun

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "feed-forward"),
                           ('activation', self.fun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b.set_value(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W.set_value(values['W'])

    def run(self, inMat):
        return self.fun(T.tensordot(inMat, self.W, axes=(2, 1)) + self.b)


class Studentise(Layer):
    """ Normal all features in batch

    :param epsilon: Stabilsation layer
    :param name: Name for layer
    """

    def __init__(self, insize, epsilon=1e-4, name="Studentise"):
        self.epsilon = epsilon
        self._insize = insize
        self._name = name

    @property
    def size(self):
        return self.insize

    def params(self):
        return []

    def json(self, params=False):
        return {'type' : "studentise"}

    def set_params(self, values):
        return

    def run(self, inMat):
        m = T.shape_padleft(T.mean(inMat, axis=(0, 1)), n_ones=2)
        v = T.shape_padleft(T.var(inMat, axis=(0, 1)), n_ones=2)
        return (inMat - m) / T.sqrt(v + self.epsilon)


class NormaliseL1(Layer):
    """ Normal all features in batch

    :param epsilon: Stabilsation layer
    :param name: Name for layer
    """

    def __init__(self, insize, epsilon=1e-4, name="Normalise"):
        self.epsilon = epsilon
        self._insize = insize
        self._name = name

    @property
    def size(self):
        return self.insize

    def params(self):
        return []

    def json(self, params=False):
        return {'type' : "normaliseL1"}

    def set_params(self, values):
        return

    def run(self, inMat):
        f = self.epsilon + T.sum(T.abs_(inMat), axis=2)
        return inMat / T.shape_padright(f)


class SoftmaxTheano(Layer):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp )

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False, name="Softmax"):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self._insize = insize
        self._size = size
        self._name = name

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "softmax"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b.set_value(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W.set_value(values['W'])

    def run(self, inMat):
        tmp = T.tensordot(inMat, self.W, axes=(2, 1)) + self.b
        out, _ = th.map(T.nnet.softmax, sequences=tmp)
        return out


class Softmax(Layer):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = row_normalise( tmp )

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False, name="Softmax"):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.W = th.shared(init((size, insize)) / np.sqrt(size + insize))
        self._insize = insize
        self._size = size
        self._name = name

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([('type', "softmax_old"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])

        if params:
            res['params'] = OrderedDict([('W', _extract(self.W)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b.set_value(values['b'])
        assert values['W'].shape == (self.size, self.insize)
        self.W.set_value(values['W'])

    def run(self, inMat):
        tmp = T.tensordot(inMat, self.W, axes=(2, 1)) + self.b
        m = T.max(tmp, axis=2, keepdims=True)
        out = T.exp(tmp - m)
        rowsum = T.sum(out, axis=2, keepdims=True)
        return out / rowsum


class Window(Layer):
    """  Create a sliding window over input

    :param w: Size of window
    :param name: Name for layer
    """

    def __init__(self, insize, w, name="Window"):
        assert w > 0, "Window size must be positive"
        assert w % 2 == 1, 'Window size should be odd'
        self.w = w
        self._insize = insize
        self._name = name

    @property
    def size(self):
        return self.w * self.insize

    def params(self):
        return []

    def json(self, params=False):
        res = OrderedDict([('type', "window")])
        if params:
            res['params'] = OrderedDict([('w', self.w)])

    def set_params(self, values):
        return

    def run(self, inMat):
        ntime, nbatch, nfeatures = T.shape(inMat)
        zeros = T.zeros((self.w // 2, nbatch, nfeatures))
        padMat = T.concatenate([zeros, inMat, zeros], axis=0)
        tmp = T.concatenate([padMat[i : 1 + i - self.w] for i in range(self.w - 1)], axis=2)
        return T.concatenate([tmp, padMat[self.w - 1 :]], axis=2)


class Convolution(Layer):
    """1D convolution over the first dimension

    Takes input of shape [time, batch, features] and produces output of shape
    [ceil((time + padding) / stride), batch, features]

    :param insize: number of features on input
    :param size: number of output features
    :param winlen: size of window over input
    :param stride: step size between successive windows
    :param init: function to initialise tensors with
    :param has_bias: whether layer has bias
    :param fun: the activation function
    :param padding_mode: str, int or (int, int)
        Controls the padding applied to the input. See conv.calculate_padding
        for available modes. Default: 'same'
    :param name: Name for layer
    """

    def __init__(self, insize, size, winlen, stride=1, init=zeros,
                 has_bias=False, fun=activation.tanh, padding_mode='same',
                 name="Convolution"):
        self._insize = insize
        self._size = size
        self._name = name
        self.winlen = winlen
        self.stride = stride
        self.fun = fun
        self.has_bias = has_bias
        self.padding_mode = padding_mode
        self.padding = conv.calculate_padding(padding_mode, winlen)

        # parameters
        fanin = insize * winlen
        fanout = (size * winlen) / float(stride)
        self.W = th.shared(init((size, insize, winlen)) /
                           np.sqrt(fanin + fanout))
        self.b = th.shared(has_bias * init(size))

    def params(self):
        return [self.W, self.b] if self.has_bias else [self.W]

    def json(self, params=False):
        res = OrderedDict([("type", "convolution"),
                           ("insize", self.insize),
                           ("size", self.size),
                           ("winlen", self.winlen),
                           ("stride", self.stride),
                           ("padding_mode", self.padding_mode),
                           ("padding", self.padding),
                           ("activation", self.fun.__name__)])
        if params:
            res['params'] = OrderedDict([("W", _extract(self.W)),
                                         ("b", _extract(self.b))])
        return res

    def set_params(self, values):
        assert values['W'].shape == (self.size, self.insize, self.winlen)
        self.W.set_value(values['W'])
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b.set_value(values['b'])

    def run(self, inMat):
        c = conv.conv_1d(inMat, self.W, self.stride, self.padding) + self.b
        return self.fun(c)


class MaxPool(Layer):
    """Max pooling over the first dimension of the input

    :param insize: dimension of input
    :param pool_size: number of elements in each pool
    :param stride: spacing between adjacent pools
    :param fun: activation function for the layer
    :param padding_mode: str, int or (int, int)
        Controls the padding applied to the input. See conv.calculate_padding
        for available modes. Default: 'same'
    :param name: Name for layer
    """

    def __init__(self, insize, pool_size, stride, fun=activation.linear,
                 padding_mode='same', name="MaxPooling"):
        self._insize = insize
        self._name = name
        self.pool_size = pool_size
        self.stride = stride
        self.fun = fun
        self.padding_mode = padding_mode
        self.padding = conv.calculate_padding(padding_mode, pool_size)

    @property
    def size(self):
        return self.insize

    def params(self):
        return []

    def json(self, params=False):
        return OrderedDict([("type", "max_pool"),
                            ("pool_size", self.pool_size),
                            ("stride", self.stride),
                            ("padding_mode", self.padding_mode),
                            ("padding", self.padding),
                            ("activation", self.fun.__name__)])

    def set_params(self, values):
        pass

    def run(self, inMat):
        p = conv.pool_1d(inMat, self.pool_size, self.stride, self.padding)
        return self.fun(p)


class Recurrent(RNN):
    """ A simple recurrent layer
        Step:  state_new = fun( [state_old, input_new] W + b )
               output_new = state_new

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, name="Recurrent"):
        self.has_bias = has_bias
        self.b = th.shared(has_bias * init(size))
        self.iW = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((size, size)) / np.sqrt(size + size))
        self.fun = fun
        self._insize = insize
        self._size = size
        self._name = name

    def params(self):
        return [self.iW, self.sW, self.b] if self.has_bias else [self.iW, self.sW]

    def json(self, params=False):
        res = OrderedDict([('type', "recurrent"),
                           ('activation', self.fun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW)),
                                         ('sW', _extract(self.sW)),
                                         ('b', _extract(self.b))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape[0] == self.size
            self.b.set_value(values['b'])
        assert values['iW'].shape == (self.size, self.insize)
        self.iW.set_value(values['iW'])
        assert values['sW'].shape == (self.size, self.size)
        self.sW.set_value(values['sW'])

    def step(self, in_vec, in_state):
        iV = T.tensordot(in_vec, self.iW, axes=(1, 1))
        sV = T.tensordot(in_state, self.sW, axes=(1, 1))
        state_out = self.fun(iV + sV + self.b)
        return state_out


class Scrn(RNN):
    """ Structurally Constrained Recurrent Network as described in
    https://arxiv.org/pdf/1412.7753.pdf (equations 4, 5 and 6)
        Step:  slow_new = (1 - a) * (input_new B) + a * slow_old
               fast_new = fun( [fast_old, slow_new, input_new] W )
               output_new = fast_new

    :param insize: Size of input to layer
    :param fast_size: Number of fast hidden units in layer
    :param slow_size: Number of slow units in hidden layer
    :param alpha: Decay coefficient for memory units
    :param init: function to initialise tensors with
    :param fun: The activation function.  Must accept a numpy array as input.
    :param name: Name for layer
    """

    def __init__(self, insize, fast_size, slow_size, init=zeros, alpha=0.95,
                 fun=activation.sigmoid, name="StructurallyConstrainedRNN"):
        # mmW is the (non-learned) memory unit decay matrix
        # the option to learn the entries of this matrix could be added later
        self.alpha = T.constant(alpha)
        self.ssW = th.shared((alpha * np.identity(slow_size)).astype(sloika_dtype))
        self.isW = th.shared(init((slow_size, insize)) / np.sqrt(slow_size + insize))
        self.sfW = th.shared(init((fast_size, slow_size)) / np.sqrt(fast_size + slow_size))
        self.ifW = th.shared(init((fast_size, insize)) / np.sqrt(fast_size + insize))
        self.ffW = th.shared(init((fast_size, fast_size)) / np.sqrt(fast_size + fast_size))
        self.fun = fun
        self._insize = insize
        self._name = name
        self.fast_size = fast_size
        self.slow_size = slow_size

    @property
    def size(self):
        return self.fast_size + self.slow_size

    def params(self):
        return [self.isW, self.sfW, self.ifW, self.ffW]

    def json(self, params=False):
        res = OrderedDict([('type', "SCRN"),
                           ('activation', self.fun.__name__),
                           ('size', self.size),
                           ('fast_size', self.fast_size),
                           ('slow_size', self.slow_size),
                           ('insize', self.insize),
                           ('alpha', float(self.alpha.get_scalar_constant_value()))])
        if params:
            res['params'] = OrderedDict([('isW', _extract(self.isW)),
                                         ('sfW', _extract(self.sfW)),
                                         ('ifW', _extract(self.ifW)),
                                         ('ffW', _extract(self.ffW))])
        return res

    def set_params(self, values):
        assert values['isW'].shape == (self.slow_size, self.insize)
        self.isW.set_value(values['isW'])
        assert values['sfW'].shape == (self.fast_size, self.slow_size)
        self.sfW.set_value(values['sfW'])
        assert values['ifW'].shape == (self.fast_size, self.insize)
        self.ifW.set_value(values['ifW'])
        assert values['ffW'].shape == (self.fast_size, self.fast_size)
        self.ffW.set_value(values['ffW'])

    def step(self, in_vec, in_state):
        in_fast = in_state[:, :self.fast_size]
        in_slow = in_state[:, self.fast_size:]
        iU = T.tensordot(in_vec, self.isW, axes=(1, 1))
        slow_out = ((1.0 - self.alpha) * iU) + (self.alpha * in_slow)
        sV = T.tensordot(slow_out, self.sfW, axes=(1, 1))
        iV = T.tensordot(in_vec, self.ifW, axes=(1, 1))
        fV = T.tensordot(in_fast, self.ffW, axes=(1, 1))
        fast_out = self.fun(sV + iV + sV)
        return T.concatenate([fast_out, slow_out], 1)


class Lstm(RNN):
    """ LSTM layer with peepholes.  Implementation is to be consistent with
    Currennt and may differ from other descriptions of LSTM networks (e.g.
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

    Step:
        v = [ input_new, output_old ]
        Pforget = gatefun( v W2 + b2 + state * p1)
        Pupdate = gatefun( v W1 + b1 + state * p0)
        Update  = fun( v W0 + b0 )
        state_new = state_old * Pforget + Update * Pupdate
        Poutput = gatefun( v W3 + b3 + state * p2)
        output_new = fun(state) * Poutput

    :Note: The inputs are arranged to maintain compatibilty it the older version
    of the LSTM layer and several of the processing steps could be optimised out.

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param has_peep: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
    mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False, has_peep=False,
                 fun=activation.tanh, gatefun=activation.sigmoid, name="LSTM"):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.has_peep = has_peep
        self.fun = fun
        self.gatefun = gatefun

        self.b = th.shared(has_bias * (init(4 * size) + np.repeat([0, 0, _FORGET_BIAS, 0], size).astype(sloika_dtype)))
        self.p = th.shared(has_peep * init((3, size)) / np.sqrt(size))
        self.iW = th.shared(init((4 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((4 * size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        if self.has_peep:
            params += [self.p]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('peep', self.has_peep)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (4, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (4, self.size, self.size))),
                                         ('b', _extract(self.b, (4, self.size))),
                                         ('p', _extract(self.p, (3, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (4, self.size)
            self.b.set_value(values['b'].transpose().reshape(-1))
        if self.has_peep:
            assert values['p'].shape == (3, self.size)
            self.p.set_value(values['p'])
        assert values['iW'].shape == (4, self.size, self.insize)
        self.iW.set_value(values['iW'].reshape((self.size * 4, self.insize)))
        assert values['sW'].shape == (4, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((self.size * 4, self.size)))

    def step(self, in_vec, in_state):
        vW = T.tensordot(in_vec, self.iW, axes=(1, 1))
        out_prev = in_state[:, :self.size]
        state = in_state[:, self.size:]
        outW = T.tensordot(out_prev, self.sW, axes=(1, 1))
        sumW = vW + outW + self.b
        sumW = sumW.reshape((-1, self.size, 4))

        #  Forget gate activation
        out_state = state * self.gatefun(sumW[:, :, 2] + state * self.p[1])
        #  Update state with input
        out_state += self.fun(sumW[:, :, 0]) * self.gatefun(sumW[:, :, 1] + state * self.p[0])
        #  Output gate activation
        out = self.fun(out_state) * self.gatefun(sumW[:, :, 3] + out_state * self.p[2])
        return T.concatenate((out, out_state), axis=1)

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat,
                         outputs_info=T.zeros((nbatch, 2 * self.size)))
        return out[:, :, :self.size]


class LstmCIFG(RNN):
    """ LSTM layer with coupled input and forget gates.

    Step:
        v = [ input_new, output_old ]
        Pforget = sigmoid( v W2 + b2 + state * p1)
        Pupdate = 1 - Pforget
        Update  = tanh( v W0 + b0 )
        state_new = state_old * Pforget + Update * Pupdate
        Poutput = sigmoid( v W3 + b3 + state * p2)
        output_new = tanh(state) * Poutput

    :Note: The inputs are arranged to maintain compatibilty it the older version
    of the LSTM layer and several of the processing steps could be optimised out.

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param has_peep: Whether layer has peep
    :param fun: The activation function.  Must accept a numpy array as input.
    :param gatefun: The activation function for gates.  Generally a monotone
    mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False, has_peep=False,
                 fun=activation.tanh, gatefun=activation.sigmoid,
                 name="CoupledLSTM"):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.has_peep = has_peep
        self.fun = fun
        self.gatefun = gatefun

        self.b = th.shared(has_bias * (init(3 * size) + np.repeat([0, _FORGET_BIAS, 0], size).astype(sloika_dtype)))
        self.p = th.shared(has_peep * init((2, size)) / np.sqrt(size))
        self.iW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((3 * size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        if self.has_peep:
            params += [self.p]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM-CIFG"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('peep', self.has_peep)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (3, self.size, self.size))),
                                         ('b', _extract(self.b, (3, self.size))),
                                         ('p', _extract(self.p, (2, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b.set_value(values['b'].transpose().reshape(-1))
        if self.has_peep:
            assert values['p'].shape == (2, self.size)
            self.p.set_value(values['p'])
        assert values['iW'].shape == (3, self.size, self.insize)
        self.iW.set_value(values['iW'].reshape((self.size * 3, self.insize)))
        assert values['sW'].shape == (3, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((self.size * 3, self.size)))

    def step(self, in_vec, in_state):
        vW = T.tensordot(in_vec, self.iW, axes=(1, 1))
        out_prev = in_state[:, :self.size]
        state = in_state[:, self.size:]
        outW = T.tensordot(out_prev, self.sW, axes=(1, 1))
        sumW = vW + outW + self.b
        sumW = sumW.reshape((-1, self.size, 3))

        #  Forget gate activation
        forget = self.gatefun(sumW[:, :, 1] + state * self.p[0])
        out_state = state * forget
        #  Update state with input
        out_state += self.fun(sumW[:, :, 0]) * (1 - forget)
        #  Output gate activation
        out = self.fun(out_state) * self.gatefun(sumW[:, :, 2] + out_state * self.p[1])
        return T.concatenate((out, out_state), axis=1)

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat,
                         outputs_info=T.zeros((nbatch, 2 * self.size)))
        return out[:, :, :self.size]


class LstmO(RNN):
    """ LSTM layer with peepholes but no output gate.

    Step:
        v = [ input_new, output_old ]
        Pforget = gatefun( v W2 + b2 + state * p1)
        Pupdate = gatefun( v W1 + b1 + state * p0)
        Update  = fun( v W0 + b0 )
        state_new = fun(state_old * Pforget + Update * Pupdate)

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param has_peep: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
    mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False, has_peep=False,
                 fun=activation.tanh, gatefun=activation.sigmoid,
                 name='TransparentLSTM'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.has_peep = has_peep
        self.fun = fun
        self.gatefun = gatefun

        self.b = th.shared(has_bias * (init(3 * size) + np.repeat([0, 0, _FORGET_BIAS], size).astype(sloika_dtype)))
        self.p = th.shared(has_peep * init((3, size)) / np.sqrt(size))
        self.iW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((3 * size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        if self.has_peep:
            params += [self.p]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM-O"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('peep', self.has_peep)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (3, self.size, self.size))),
                                         ('b', _extract(self.b, (3, self.size))),
                                         ('p', _extract(self.p, (3, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b.set_value(values['b'].reshape(-1))
        if self.has_peep:
            assert values['p'].shape == (3, self.size)
            self.p.set_value(values['p'])
        assert values['iW'].shape == (3, self.size, self.insize)
        self.iW.set_value(values['iW'].reshape((3 * self.size, self.insize)))
        assert values['sW'].shape == (3, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((3 * self.size, self.size)))

    def step(self, in_vec, in_state):
        vW = T.tensordot(in_vec, self.iW, axes=(1, 1))
        outW = T.tensordot(in_state, self.sW, axes=(1, 1))
        sumW = vW + outW + self.b
        sumW = sumW.reshape((-1, 3, self.size))

        #  Forget gate activation
        state = in_state * self.gatefun(sumW[:, 2] + in_state * self.p[2])
        #  Update state with input
        state += self.fun(sumW[:, 0] + in_state * self.p[0]) * self.gatefun(sumW[:, 1] + in_state * self.p[1])
        return state


class Forget(RNN):
    """ Simple forget gate

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
    mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid,
                 name='ForgetfulRNN'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun

        self.b = th.shared(has_bias * (init(2 * size) + np.repeat([_FORGET_BIAS, 0], size).astype(sloika_dtype)))
        self.iW = th.shared(init((2 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((2 * size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.iW, self.sW]
        if self.has_bias:
            params += [self.b]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "forget gate"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (2, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (2, self.size, self.size))),
                                         ('b', _extract(self.b, (2, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (2, self.size)
            self.b.set_value(values['b'].reshape(-1))
        assert values['iW'].shape == (2, self.size, self.insize)
        self.iW.set_value(values['iW'].reshape((2 * self.size, self.insize)))
        assert values['sW'].shape == (2, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((2 * self.size, self.size)))

    def step(self, in_vec, in_state):
        vI = T.tensordot(in_vec, self.iW, axes=(1, 1))
        vS = T.tensordot(in_state, self.sW, axes=(1, 1))
        vT = vI + vS + self.b
        vT = vT.reshape((-1, 2, self.size))

        forget = self.gatefun(vT[:, 0])
        state = in_state * forget + (1.0 - forget) * self.fun(vT[:, 1])
        return state


class Gru(RNN):
    """ Gated Recurrent Unit

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
    mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid, name='GRU'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun = gatefun

        self.b = th.shared(has_bias * init(3 * size))
        self.iW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((2 * size, size)) / np.sqrt(size + size))
        self.sW2 = th.shared(init((size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.iW, self.sW, self.sW2]
        if self.has_bias:
            params += [self.b]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "GRU"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _extract(self.iW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (2, self.size, self.size))),
                                         ('sW2', _extract(self.sW2)),
                                         ('b', _extract(self.b, (3, self.size)))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b.set_value(values['b'].reshape(-1))
        assert values['iW'].shape == (3, self.size, self.insize)
        self.iW.set_value(values['iW'].reshape((3 * self.size, self.insize)))
        assert values['sW'].shape == (2, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((2 * self.size, self.size)))
        assert values['sW2'].shape == (self.size, self.size)
        self.sW2.set_value(values['sW2'])

    def step(self, in_vec, in_state):
        vI = T.tensordot(in_vec, self.iW, axes=(1, 1)) + self.b
        vS = T.tensordot(in_state, self.sW, axes=(1, 1))
        vT = vI[:, :2 * self.size] + vS
        vT = vT.reshape((-1, 2, self.size))

        z = self.gatefun(vT[:, 0])
        r = self.gatefun(vT[:, 1])
        y = T.tensordot(r * in_state, self.sW2, axes=(1, 1))
        hbar = self.fun(vI[:, 2 * self.size:] + y)
        state = z * in_state + (1 - z) * hbar
        return state


class Mut1(RNN):
    """ Based on MUT1 from Jozefowicz

    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    However, MutN as described expects scalar inputs, whereas we may have
    insize > 1. Where Josefowicz applied a non-linearity to a scaler input, we
    replace this with a feed-forward layer u.

    Step:
        u = fun(x W_xu + B_u)
        r = gatefun(x W_xr + state W_hr + b_r)
        z = gatefun(x W_xz + b_z)
        state_new = fun((r * state) W_hh + u + b_h) * z + state * (1 - z)

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
        mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid, name='MUT1'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun = gatefun

        self.b_z = th.shared(has_bias * (init(size) + _FORGET_BIAS).astype(sloika_dtype))
        self.b_r = th.shared(has_bias * init(size))
        self.b_h = th.shared(has_bias * init(size))
        self.b_u = th.shared(has_bias * init(size))
        self.W_xu = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_xz = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_xr = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_hr = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_hh = th.shared(init((size, size)) / np.sqrt(size + size))

    def params(self):
        params = [self.W_xu, self.W_xz, self.W_xr, self.W_hr, self.W_hh]
        if self.has_bias:
            params += [self.b_u, self.b_r, self.b_z, self.b_h]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "MUT1"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W_xu', _extract(self.W_xu)),
                                         ('W_xz', _extract(self.W_xz)),
                                         ('W_xr', _extract(self.W_xr)),
                                         ('W_hr', _extract(self.W_hr)),
                                         ('W_hh', _extract(self.W_hh)),
                                         ('b_u', _extract(self.b_u)),
                                         ('b_z', _extract(self.b_z)),
                                         ('b_h', _extract(self.b_h)),
                                         ('b_r', _extract(self.b_r))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b_u'].shape == (self.size)
            self.b_u.set_value(values['b_u'])
            assert values['b_r'].shape == (self.size)
            self.b_u.set_value(values['b_r'])
            assert values['b_h'].shape == (self.size)
            self.b_u.set_value(values['b_h'])
            assert values['b_z'].shape == (self.size)
            self.b_u.set_value(values['b_z'])
        assert values['W_xu'].shape == (self.size, self.insize)
        self.W_xu.set_value(values['W_xu'])
        assert values['W_xz'].shape == (self.size, self.insize)
        self.W_xz.set_value(values['W_xz'])
        assert values['W_xr'].shape == (self.size, self.insize)
        self.W_xr.set_value(values['W_xr'])
        assert values['W_hr'].shape == (self.size, self.size)
        self.W_hr.set_value(values['W_hr'])
        assert values['W_hh'].shape == (self.size, self.size)
        self.W_hh.set_value(values['W_hh'])

    def step(self, in_vec, in_state):
        u = self.fun(T.tensordot(in_vec, self.W_xu, axes=(1, 1)) + self.b_u)
        z = self.gatefun(T.tensordot(in_vec, self.W_xz, axes=(1, 1)) + self.b_z)
        r = self.gatefun(T.tensordot(in_vec, self.W_xr, axes=(1, 1)) +
                         T.tensordot(in_state, self.W_hr, axes=(1, 1)) + self.b_r)
        y = T.tensordot(r * in_state, self.W_hh, axes=(1, 1))
        state = self.fun(y + u + self.b_h) * z + (1 - z) * in_state
        return state


class Mut2(RNN):
    """ MUT2 from Jozefowicz

    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    However, MutN as described expects scalar inputs, whereas we may have
    insize > 1. Where Josefowicz applied a non-linearity to a scaler input, we
    replace this with a feed-forward layer u.

    Step:
        u = fun(x W_xu + B_u)
        r = gatefun(u + state W_hr + b_r)
        z = gatefun(x W_xz + state W_hz + b_z)
        state_new = fun((r * state) W_hh + x W_xh + b_h) * z + state * (1 - z)

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    :param gatefun: The activation function for gates.  Generally a monotone
        mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid, name='MUT2'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun = gatefun

        self.b_z = th.shared(has_bias * (init(size) + _FORGET_BIAS).astype(sloika_dtype))
        self.b_r = th.shared(has_bias * init(size))
        self.b_h = th.shared(has_bias * init(size))
        self.b_u = th.shared(has_bias * init(size))
        self.W_xu = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_xz = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_hz = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_hr = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_hh = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_xh = th.shared(init((size, insize)) / np.sqrt(size + size))

    def params(self):
        params = [self.W_xu, self.W_xz, self.W_hz, self.W_hr, self.W_hh, self.W_xh]
        if self.has_bias:
            params += [self.b_u, self.b_r, self.b_z, self.b_h]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "MUT2"),
                           ('activation', self.fun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W_xu', _extract(self.W_xu)),
                                         ('W_xz', _extract(self.W_xz)),
                                         ('W_hz', _extract(self.W_hz)),
                                         ('W_hr', _extract(self.W_hr)),
                                         ('W_hh', _extract(self.W_hh)),
                                         ('W_xh', _extract(self.W_xh)),
                                         ('b_u', _extract(self.b_u)),
                                         ('b_z', _extract(self.b_z)),
                                         ('b_h', _extract(self.b_h)),
                                         ('b_r', _extract(self.b_r))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b_u'].shape == (self.size)
            self.b_u.set_value(values['b_u'])
            assert values['b_r'].shape == (self.size)
            self.b_u.set_value(values['b_r'])
            assert values['b_h'].shape == (self.size)
            self.b_u.set_value(values['b_h'])
            assert values['b_z'].shape == (self.size)
            self.b_u.set_value(values['b_z'])
        assert values['W_xu'].shape == (self.size, self.insize)
        self.W_xu.set_value(values['W_xu'])
        assert values['W_xz'].shape == (self.size, self.insize)
        self.W_xz.set_value(values['W_xz'])
        assert values['W_xh'].shape == (self.size, self.insize)
        self.W_xh.set_value(values['W_xh'])
        assert values['W_hr'].shape == (self.size, self.size)
        self.W_hr.set_value(values['W_hr'])
        assert values['W_hh'].shape == (self.size, self.size)
        self.W_hh.set_value(values['W_hh'])
        assert values['W_hz'].shape == (self.size, self.size)
        self.W_hz.set_value(values['W_hz'])

    def step(self, in_vec, in_state):
        u = self.fun(T.tensordot(in_vec, self.W_xu, axes=(1, 1)) + self.b_u)
        z = self.gatefun(T.tensordot(in_vec, self.W_xz, axes=(1, 1)) +
                         T.tensordot(in_state, self.W_hz, axes=(1, 1)) + self.b_z)
        r = self.gatefun(u + T.tensordot(in_state, self.W_hr, axes=(1, 1)) + self.b_r)
        y = T.tensordot(r * in_state, self.W_hh, axes=(1, 1))
        v = T.tensordot(in_vec, self.W_xh, axes=(1, 1))
        state = self.fun(y + v + self.b_h) * z + (1 - z) * in_state
        return state


class Mut3(RNN):
    """ Based on MUT3 from Jozefowicz

    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    However, MutN as described expects scalar inputs, whereas we may have
    insize > 1. Where Josefowicz applied a non-linearity to a scaler input, we
    replace this with a feed-forward layer u.

    Step:
        u = fun(x W_xu + B_u)
        r = gatefun(x W_xr + state W_hr + b_r)
        z = gatefun(x W_xz + fun(state) W_hz + b_z)
        state_new = fun((r * state) W_hh + x W_xh + b_h) * z + state * (1 - z)

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.  Must accept a numpy array as input.
    :param gatefun: The activation function for gates.  Generally a monotone
        mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid, name='MUT3'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun = gatefun

        self.b_z = th.shared(has_bias * (init(size) + _FORGET_BIAS).astype(sloika_dtype))
        self.b_r = th.shared(has_bias * init(size))
        self.b_h = th.shared(has_bias * init(size))
        self.b_u = th.shared(has_bias * init(size))
        self.W_xu = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_xz = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_hz = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_xr = th.shared(init((size, insize)) / np.sqrt(insize + size))
        self.W_hr = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_hh = th.shared(init((size, size)) / np.sqrt(size + size))
        self.W_xh = th.shared(init((size, insize)) / np.sqrt(size + size))

    def params(self):
        params = [self.W_xu, self.W_xz, self.W_hz, self.W_xr, self.W_hr, self.W_hh, self.W_xh]
        if self.has_bias:
            params += [self.b_u, self.b_r, self.b_z, self.b_h]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "MUT3"),
                           ('activation', self.fun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W_xu', _extract(self.W_xu)),
                                         ('W_xz', _extract(self.W_xz)),
                                         ('W_hz', _extract(self.W_hz)),
                                         ('W_xr', _extract(self.W_xr)),
                                         ('W_hr', _extract(self.W_hr)),
                                         ('W_hh', _extract(self.W_hh)),
                                         ('W_xh', _extract(self.W_xh)),
                                         ('b_u', _extract(self.b_u)),
                                         ('b_z', _extract(self.b_z)),
                                         ('b_h', _extract(self.b_h)),
                                         ('b_r', _extract(self.b_r))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b_u'].shape == (self.size)
            self.b_u.set_value(values['b_u'])
            assert values['b_r'].shape == (self.size)
            self.b_u.set_value(values['b_r'])
            assert values['b_h'].shape == (self.size)
            self.b_u.set_value(values['b_h'])
            assert values['b_z'].shape == (self.size)
            self.b_u.set_value(values['b_z'])
        assert values['W_xu'].shape == (self.size, self.insize)
        self.W_xu.set_value(values['W_xu'])
        assert values['W_xz'].shape == (self.size, self.insize)
        self.W_xz.set_value(values['W_xz'])
        assert values['W_xh'].shape == (self.size, self.insize)
        self.W_xh.set_value(values['W_xh'])
        assert values['W_xr'].shape == (self.size, self.insize)
        self.W_xr.set_value(values['W_xr'])
        assert values['W_hr'].shape == (self.size, self.size)
        self.W_hr.set_value(values['W_hr'])
        assert values['W_hh'].shape == (self.size, self.size)
        self.W_hh.set_value(values['W_hh'])
        assert values['W_hz'].shape == (self.size, self.size)
        self.W_hz.set_value(values['W_hz'])

    def step(self, in_vec, in_state):
        u = self.fun(T.tensordot(in_vec, self.W_xu, axes=(1, 1)) + self.b_u)
        z = self.gatefun(T.tensordot(in_vec, self.W_xz, axes=(1, 1)) +
                         T.tensordot(self.fun(in_state), self.W_hz, axes=(1, 1)) + self.b_z)
        r = self.gatefun(T.tensordot(in_vec, self.W_xr, axes=(1, 1)) +
                         T.tensordot(in_state, self.W_hr, axes=(1, 1)) + self.b_r)
        y = T.tensordot(r * in_state, self.W_hh, axes=(1, 1))
        v = T.tensordot(in_vec, self.W_xh, axes=(1, 1))
        state = self.fun(y + v + self.b_h) * z + (1 - z) * in_state
        return state


class Genmut(RNN):
    """ Generalisation of MUT1 from Jozefowicz

    http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    However, MutN as described expects scalar inputs, whereas we may have
    insize > 1. Where Josefowicz applied a non-linearity to a scaler input, we
    replace this with a feed-forward layer u.

    Step:
        u = fun(x W_xu + state W_hu + b_u)
        r = gatefun(x W_xr + state W_hr + b_r)
        z = gatefun(x W_xz + state W_hz + b_z)
        state_new = fun((r * state) W_hh + u + b_h) * z + state * (1 - z)

    :param insize: Size of input to layer
    :param size: Layer size
    :param init: function to initialise tensors with
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    :param gatefun: The activation function for gates.  Generally a monotone
        mapping from (-inf, inf) -> [0, 1]
    :param name: Name for layer
    """

    def __init__(self, insize, size, init=zeros, has_bias=False,
                 fun=activation.tanh, gatefun=activation.sigmoid,
                 name='GeneralisedMUT1'):
        self._size = size
        self._insize = insize
        self._name = name
        self.has_bias = has_bias
        self.fun = fun
        self.gatefun = gatefun

        self.b = th.shared(has_bias * init(3 * size))
        self.xW = th.shared(init((3 * size, insize)) / np.sqrt(insize + size))
        self.sW = th.shared(init((3 * size, size)) / np.sqrt(size + size))
        self.sW2 = th.shared(init((size, size)) / np.sqrt(size + size))
        self.b2 = th.shared(has_bias * init(size))

    def params(self):
        params = [self.xW, self.sW, self.sW2]
        if self.has_bias:
            params += [self.b, self.b2]
        return params

    def json(self, params=False):
        res = OrderedDict([('type', "Genmut"),
                           ('activation', self.fun.__name__),
                           ('gate', self.gatefun.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('xW', _extract(self.xW, (3, self.size, self.insize))),
                                         ('sW', _extract(self.sW, (3, self.size, self.size))),
                                         ('sW2', _extract(self.sW2)),
                                         ('b', _extract(self.b, (3, self.size))),
                                         ('b2', _extract(self.b2))])
        return res

    def set_params(self, values):
        if self.has_bias:
            assert values['b'].shape == (3, self.size)
            self.b.set_value(values['b'].reshape(-1))
            assert values['b2'].shape == (self.size)
            self.b2.set_value(values['b2'])
        assert values['xW'].shape == (3, self.size, self.insize)
        self.xW.set_value(values['xW'].reshape((3 * self.size, self.insize)))
        assert values['sW'].shape == (3, self.size, self.size)
        self.sW.set_value(values['sW'].reshape((3 * self.size, self.size)))
        assert values['sW2'].shape == (self.size, self.size)
        self.sW2.set_value(values['sW2'])

    def step(self, in_vec, in_state):
        iT = (T.tensordot(in_vec, self.xW, axes=(1, 1)) + T.tensordot(in_state, self.sW, axes=(1, 1)) + self.b)
        iT = iT.reshape((-1, 3, self.size))

        u = self.fun(iT[:, 0])
        r = self.gatefun(iT[:, 1])
        z = self.gatefun(iT[:, 2])
        y = T.tensordot(r * in_state, self.sW2, axes=(1, 1))
        state = self.fun(y + u + self.b2) * z + (1 - z) * in_state
        return state


class Reverse(Layer):
    """  Runs a recurrent layer in reverse time (backwards)

    :param layer: A :class:`layer` to reverse
    :param name: Name for layer
    """

    def __init__(self, layer, name='Reverse'):
        self.layer = layer
        self._name = name

    @property
    def insize(self):
        return self.layer.insize

    @property
    def size(self):
        return self.layer.size

    def params(self):
        return self.layer.params()

    def json(self, params=False):
        return OrderedDict([('type', "reverse"),
                            ('sublayer', self.layer.json(params))])

    def set_params(self, values):
        return

    def run(self, inMat):
        return self.layer.run(inMat[::-1])[::-1]


class Parallel(Layer):
    """ Run multiple layers in parallel (all have same input and outputs are concatenated)

    :param layers: A list of :class:`layer` to run in parallel
    :param name: Name for layer
    """

    def __init__(self, layers, name='Parallel'):
        assert len(layers) > 0, "A Parallel layer cannot be empty"
        self.layers = layers
        self._name = name
        insize = self.layers[0].insize
        is_consistent = all(x.insize == self.insize for x in self.layers)
        assert is_consistent, "Parallel layer has inconsistent sizes"

    @property
    def insize(self):
        return self.layers[0].insize

    @property
    def size(self):
        return sum(x.size for x in self.layers)

    def params(self):
        return reduce(lambda x, y: x + y.params(), self.layers, [])

    def json(self, params=False):
        return OrderedDict([('type', "parallel"),
                            ('sublayers', [layer.json(params) for layer in self.layers])])

    def set_params(self, values):
        return

    def run(self, inMat):
        return T.concatenate([x.run(inMat) for x in self.layers], axis=2)


class Residual(Layer):
    """ Wraps a layer with a residual connection

    :param layer: A :class:`Layer` to wrap
    :param name: Name for layer
    """

    def __init__(self, layer, name='Residual'):
        assert layer.insize == layer.size, "Residual connections only supported when input and output sizes are equal"
        self.layer = layer
        self._name = name

    @property
    def insize(self):
        return self.layer.insize

    @property
    def size(self):
        return self.layer.size

    def params(self):
        return self.layer.params

    def json(self, params=False):
        return OrderedDict([('type', "residual"),
                            ('sublayer', self.layer.json(params))])

    def set_params(self, values):
        return

    def run(self, inMat):
        return inMat + self.layer.run(inMat)


class Serial(Layer):
    """ Run multiple layers serially: output of a layer is the input for the next layer

    :param layers: A list of :class:`layer` to run in series
    :param name: Name for layer
    """

    def __init__(self, layers, name='Serial'):
        assert len(layers) > 0, "A Serial layer cannot be empty"
        self.layers = layers
        self._name = name
        is_consistent = all(x.size == y.insize for x, y in zip(layers, layers[1:]))
        assert is_consistent, "Serial layer has inconistent sizes"

    @property
    def insize(self):
        return self.layers[0].insize

    @property
    def size(self):
        return self.layers[-1].size

    def params(self):
        return reduce(lambda x, y: x + y.params(), self.layers, [])

    def json(self, params=False):
        return OrderedDict([('type', "serial"),
                            ('sublayers', [layer.json(params) for layer in self.layers])])

    def set_params(self, values):
        return

    def run(self, inMat):
        tmp = inMat
        for layer in self.layers:
            tmp = layer.run(tmp)
        return tmp


class Decode(RNN):
    """ Forward pass of a Viterbi decoder

    :param name: Name for layer
    """

    def __init__(self, k, name='ForwardsViterbi'):
        self._NBASE = T.constant(DEFAULT_NBASE, dtype='int32')
        self._NSTEP = T.constant(_NSTEP, dtype='int32')
        self._NSKIP = T.constant(_NSKIP, dtype='int32')
        self.size = T.constant(nkmer(k), dtype='int32')
        self.rstep = T.constant(DEFAULT_NBASE ** (k - 1), dtype='int32')
        self.rskip = T.constant(DEFAULT_NBASE ** (k - 2), dtype='int32')
        self._name = name

    def params(self):
        return []

    def json(self, params=False):
        return OrderedDict([('type', "decode")])

    def set_params(self, values):
        return

    def step(self, in_vec, in_state):
        pscore = in_state[:, :self.size]
        # Stay
        score = pscore
        iscore = T.zeros_like(score)
        iscore += T.arange(0.0, stop=self.size)
        # Step
        pscore = pscore.reshape((-1, self._NSTEP, self.rstep))
        score2 = T.repeat(T.max(pscore, axis=1), self._NSTEP)
        iscore2 = T.repeat(self.rstep * T.argmax(pscore, axis=1) + T.arange(0.0,
                                                                            stop=self.rstep,
                                                                            dtype=sloika_dtype), self._NSTEP)
        iscore2 = iscore2.reshape((-1, self.size))
        score2 = score2.reshape((-1, self.size))
        iscore = T.switch(T.gt(score, score2), iscore, iscore2)
        score = T.maximum(score, score2)
        # Skip
        pscore = pscore.reshape((-1, self._NSKIP, self.rskip))
        score2 = T.repeat(T.max(pscore, axis=1), self._NSKIP)
        iscore2 = T.repeat(self.rstep * T.argmax(pscore, axis=1) + T.arange(0.0, stop=self.rskip), self._NSKIP)
        iscore2 = iscore2.reshape((-1, self.size))
        score2 = score2.reshape((-1, self.size))
        iscore = T.switch(T.gt(score, score2), iscore, iscore2)
        score = T.maximum(score, score2)

        score += T.log(T.nnet.softmax(in_vec))
        return T.concatenate((iscore, score), axis=1)

    def run(self, inMat):
        nbatch = T.shape(inMat)[1]
        out, _ = th.scan(self.step, sequences=inMat,
                         outputs_info=T.zeros((nbatch, 2 * self.size)))
        return out[:, :, self.size]


def birnn(forward, backward, name='BiRNN'):
    """  Creates a bidirectional RNN from two RNNs

    :param forward: A :class:`layer` to run forwards
    :param backward: A :class:`layer` to run backwards
    :param name: Name for layer
    """
    return Parallel([forward, Reverse(backward)], name=name)
