import sloika.module_tools as smt


def network(klen, sd, nbase=smt.DEFAULT_NBASE, nfeature=1, winlen=11, stride=5):
    """ Create a network with convolution input layer and five alternating-in-direction GRU layers

    :param klen: Length of kmer
    :param sd: Standard Deviation of initialisation noise
    :param nbase: Number of distinct bases
    :param nfeature: Number of features per time-step
    :param winlen: Length of window over data
    :param stride: Stride over data

    :returns: a `class`:layer.Layer:
    """

    n = 96
    fun = smt.tanh
    init = smt.partial(smt.truncated_normal, sd=sd)

    return smt.Serial([smt.Convolution(nfeature, n, winlen, stride, init=init, has_bias=True, fun=smt.elu),

                       smt.Reverse(smt.Gru(n, n, init=init, has_bias=True, fun=fun)),

                       smt.Gru(n, n, init=init, has_bias=True, fun=fun),

                       smt.Reverse(smt.Gru(n, n, init=init, has_bias=True, fun=fun)),

                       smt.Gru(n, n, init=init, has_bias=True, fun=fun),

                       smt.Reverse(smt.Gru(n, n, init=init, has_bias=True, fun=fun)),

                       smt.Softmax(n, smt.nstate(klen, nbase=nbase), init=init, has_bias=True)

                       ])
