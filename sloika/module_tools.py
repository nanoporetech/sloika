from functools import partial
from scipy.stats import truncnorm
from sloika.config import sloika_dtype
from sloika.activation import *
from sloika.layers import *
from sloika.variables import *


def truncated_normal(size, sd):
    ''' Truncated normal for Xavier style initiation
    '''
    res = sd * truncnorm.rvs(-2, 2, size=size)
    return res.astype(sloika_dtype)
