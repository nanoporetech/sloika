"""Collection of function decorators."""
import sys
from functools import partial
import traceback


def docstring_parameter(*sub):
    """Allow docstrings to contain parameters."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


def _try_except_pass(func, *args, **kwargs):
    """Implementation of try_except_pass below. When wrapping a function we
    would ordinarily form a closure over a (sub)set of the inputs. Such
    closures cannot be pickled however since the wrapper name is not
    importable. We get around this by using functools.partial (which is
    pickleable). The result is that we can decorate a function to mask
    exceptions thrown by it.
    """

    # Strip out "our" arguments, this slightly perverse business allows
    #    us to call the target function with multiple arguments.
    recover = kwargs.pop('recover', None)
    recover_fail = kwargs.pop('recover_fail', False)
    try:
        return func(*args, **kwargs)
    except:
        exc_info = sys.exc_info()
        try:
            if recover is not None:
                recover(*args, **kwargs)
        except Exception as e:
            sys.stderr.write("Unrecoverable error.")
            if recover_fail:
                raise e
            else:
                traceback.print_exc(sys.exc_info()[2])
        # print the original traceback
        traceback.print_tb(exc_info[2])
        return None


def try_except_pass(func, recover=None, recover_fail=False):
    """Wrap a function to mask exceptions that it may raise. This is
    equivalent to::

        def try_except_pass(func):
            def wrapped()
                try:
                    func()
                except Exception as e:
                    print str(e)
            return wrapped

    in the simplest sense, but the resulting function can be pickled.

    :param func: function to call
    :param recover: function to call immediately after exception thrown in
        calling `func`. Will be passed same args and kwargs as `func`.
    :param recover_fail: raise exception if recover function raises?

    ..note::
        See `_try_except_pass` for implementation, which is not locally
        scoped here because we wish for it to be pickleable.

    ..warning::
        Best practice would suggest this to be a dangerous function. Consider
        rewriting the target function to better handle its errors. The use
        case here is intended to be ignoring exceptions raised by functions
        when mapped over arguments, if failures for some arguments can be
        tolerated.

    """
    return partial(_try_except_pass, func, recover=recover, recover_fail=recover_fail)
