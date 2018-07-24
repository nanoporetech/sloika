from multiprocessing import Process
from multiprocessing import SimpleQueue
import pickle
import shutil
import sys
import tempfile
import warnings


def _compile_model(outqueue, model_file, output_file=None):
    """  Compile network if necessary

    Where the network is already compiled, a temporary copy
    is created.

    :param outqueue: Queue to output filename
    :param model_file: File to read network from
    :param output_file: File to output to.  If None, generate a filename

    :returns: places name of a file containined compiled model into queue
    """
    from sloika import layers
    import theano
    import logging

    logging.getLogger("theano.gof.compilelock").setLevel(logging.WARNING)

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode='wb', dir='', suffix='.pkl', delete=False) as fh:
            output_file = fh.name

    sys.setrecursionlimit(10000)
    try:
        with open(model_file, 'rb') as fh:
            network = pickle.load(fh)
    except UnicodeDecodeError:
        with open(model_file, 'rb') as fh:
            network = pickle.load(fh, encoding='latin1')
            warnings.warn("Support for python 2 pickles will be dropped: {}".format(model_file))
    if isinstance(network, layers.Layer):
        #  File contains network to compile
        with open(output_file, 'wb') as fh:
            compiled_network = network.compile()
            pickle.dump(compiled_network, fh, protocol=pickle.HIGHEST_PROTOCOL)
    elif isinstance(network, theano.compile.function_module.Function):
        #  Network is already compiled - make temporary copy
        shutil.copy(model_file, output_file)
    else:
        sys.exit(1)

    outqueue.put(output_file)


def compile_model(model_file, output_file=None):
    """  Compile network in separate thread

    To avoid initialising Theano in main thread, compilation must be done in a
    separate process. The reason for avoidance is that forking a process after
    CUDA context is initialised is not supported, so, if this is a multiprocess
    run, processes must be created before importing theano.sandbox.cuda.

    Where the network is already compiled, a temporary copy is created.

    :param model_file: File to read network from
    :param output_file: File to output to.  If None, generate a filename

    :returns: A filename containing a compiled network.
    """
    queue = SimpleQueue()
    p = Process(target=_compile_model, args=(queue, model_file, output_file))
    p.start()
    p.join()
    if p.exitcode != 0:
        output_file = None
        raise ValueError("Model file {} was neither a network nor compiled network".format(model_file))
    else:
        output_file = queue.get()

    return output_file
