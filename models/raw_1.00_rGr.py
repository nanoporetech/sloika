from sloika.module_tools import *


def network(klen, sd, nbase=DEFAULT_NBASE, nfeature=1, winlen=11, stride=2):

    n = 128
    k = 110
    l = 142
    m = 110
    fun = tanh
    init = partial(truncated_normal, sd=sd)

    return Serial([Convolution(nfeature, n, winlen, stride, init=init, has_bias=True, fun=fun),

                   Reverse(Gru(n, k, init=init, has_bias=True, fun=fun)),

                   Gru(k, l, init=init, has_bias=True, fun=fun),

                   Reverse(Gru(l, m, init=init, has_bias=True, fun=fun)),

                   Softmax(m, nstate(klen, nbase=nbase), init=init, has_bias=True)

                   ])
