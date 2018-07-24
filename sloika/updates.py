from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T

from sloika.config import sloika_dtype


def sgd(network, loss, rate, momentum, clip=5.0):
    """  Stochastic Gradient Descent with momentum

    :param network: network to optimise
    :param loss: loss function to optimise over
    :param rate: rate (step size) for SGD
    :param momentum: momentum (decay for previous steps)

    :returns: a dictionary containing update functions for Tensors
    """
    assert momentum >= 0, "Momentum for SGD must be non-negative"

    params = network.params()
    updates = OrderedDict()
    gradients = th.grad(loss, params)
    for param, grad in zip(params, gradients):
        val = param.get_value(borrow=True)
        vel = th.shared(np.zeros(val.shape, dtype=val.dtype))

        grad_clip = T.clip(grad, -clip, clip)

        updates[vel] = momentum * vel - rate * grad_clip
        updates[param] = param + updates[vel]

    return updates


def adam(network, loss, rate, decay, epsilon=1e-8, clip=5.0, mrate=0.0005):
    """  ADAMski optimiser

    Similar to ADAM optimizer but with momentum phased in gradually from 0,
    as having lower momentum at the start of training seems to be beneficial.
    See: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf page 10

    :param network: network to optimise
    :param loss: loss function to optimise over
    :param rate: rate (step size) for optimiser
    :param decay: decay for estimate of gradient and curvature
    :param epsilon: same parameter to prevent reciprocal of variance exploding
    :param mrate: Rate at which momentum is increased.  None = ADAM optimiser

    :returns: a dictionary containing update functions for Tensors
    """
    assert decay > (0.0, 0.0), "Decay must be non-negative"
    assert decay < (1.0, 1.0), "Decay must be less-than or equal to one"
    assert mrate is None or mrate > 0.0, "Rate of momentum increase must be positive"
    if mrate is not None:
        _M_RATE = -np.float_(mrate).astype(sloika_dtype)
        _M_P = np.exp(_M_RATE)
        _M_K = (1.0 - decay[0]) * decay[0] * _M_P / (1.0 - _M_P * decay[0])
        _M_K = np.float_(_M_K).astype(sloika_dtype)
    else:
        _M_RATE = -np.float_(1e30).astype(sloika_dtype)
        _M_P = np.float_(0.0).astype(sloika_dtype)
        _M_K = np.float_(0.0).astype(sloika_dtype)

    params = network.params()
    updates = OrderedDict()
    gradients = th.grad(loss, params)

    ldecay = np.log(decay, dtype=sloika_dtype)

    t = th.shared(np.float32(0.0).astype(sloika_dtype))
    lr_t = th.shared(np.float32(0.0).astype(sloika_dtype))
    momentum_decay = th.shared(np.float32(0.0).astype(sloika_dtype))
    updates[t] = t + 1.0
    momentum_factor = _M_K * T.expm1(t * (ldecay[0] + _M_RATE)) - T.expm1(updates[t] * ldecay[0])
    updates[lr_t] = rate * T.sqrt(-T.expm1(updates[t] * ldecay[1])) / momentum_factor
    updates[momentum_decay] = -decay[0] * T.expm1(updates[t] * _M_RATE)
    for param, grad in zip(params, gradients):
        val = param.get_value(borrow=True)
        momentum = th.shared(np.zeros(val.shape, dtype=val.dtype))
        variance = th.shared(np.zeros(val.shape, dtype=val.dtype))

        grad_clip = T.clip(grad, -clip, clip)

        updates[momentum] = updates[momentum_decay] * momentum + (1.0 - decay[0]) * grad_clip
        updates[variance] = decay[1] * variance + (1.0 - decay[1]) * T.sqr(grad_clip)
        updates[param] = param - updates[lr_t] * updates[momentum] / (T.sqrt(updates[variance]) + epsilon)

    return updates


def param_sqr(network):
    """  Return sum of squares of network parameters

    :param network:  Network of which parmeters to sum

    :returns: A :tensor:`scalar` representing the sum of squares
    """
    params = network.params()
    psum = T.sum(T.sqr(params[0]))
    for param in params[1:]:
        psum += T.sum(T.sqr(param))
    return psum
