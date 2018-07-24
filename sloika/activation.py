import theano.tensor as T
#  Some activation functions
#  Many based on M-estimations functions, see
#  http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html


#  Unbounded
def linear(x):
    return x


def relu(x):
    return T.nnet.relu(x)


def relu_smooth(x):
    y = T.clip(x, 0.0, 1.0)
    return T.square(y) - 2.0 * y + x + T.abs_(x)


def softplus(x):
    """  Softplus function log(1 + exp(x))

        Calculated in a way stable to large and small values of x.  The version
        of this routine in theano.tensor.nnet clips the range of x, potential
        causing NaN's to occur in the softmax (all inputs clipped to zero).

        x >=0  -->  x + log1p(exp(-x))
        x < 0  -->  log1p(exp(x))

        This is equivalent to relu(x) + log1p(exp(-|x|))
    """
    absx = T.abs_(x)
    softplus_neg = T.log1p(T.exp(-absx))
    return relu(x) + softplus_neg


def elu(x):
    """  Exponential Linear Unit
         See https://arxiv.org/pdf/1511.07289.pdf
    """
    return T.switch(x > 0, x, T.expm1(x))


def exp(x):
    return T.exp(x)


#  Bounded and monotonic


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def erf(x):
    return T.erf(x)


def L1mL2(x):
    return x / T.sqrt(1.0 + 0.5 * T.sqr(x))


def fair(x):
    return x / (1.0 + T.abs_(x) / 1.3998)


def retu(x):
    """ Rectifying activation followed by Tanh

    Inspired by more biological neural activation, see figure 1
    http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf
    """
    return T.tanh(T.nnet.relu(x))


def tanh_pm(x):
    """  Poor man's tanh
    Linear approximation by tangent at x=0.  Clip into valid range.
    """
    return T.clip(x, -1.0, 1.0)


def sigmoid_pm(x):
    """ Poor man's sigmoid
    Linear approximation by tangent at x=0.  Clip into valid range.
    """
    return T.clip(0.5 + 0.25 * x, 0.0, 1.0)


def bounded_linear(x):
    """ Linear activation clipped into -1, 1
    """
    return T.clip(x, -1.0, 1.0)


#  Bounded and redescending
def sin(x):
    return T.sin(x)


def cauchy(x):
    return x / (1.0 + T.sqr(x / 2.3849))


def geman_mcclure(x):
    return x / T.sqr(1.0 + T.sqr(x))


def welsh(x):
    return x * T.exp(-T.sqr(x / 2.9846))
