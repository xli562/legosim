import os
import sys
from math import isqrt
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger

def factor_pairs(n:int) -> list:
    """ Returns 2-number factorizations of the input.
    
    :param n: The non-negative integer to be factorized

    Examples
    --------
    >>> factor_pairs(7)
    [[1, 7]]
    >>> factor_pairs(36)
    [[6, 6], [4, 9], [3, 12], [2, 18], [1, 36]]
    >>> factor_pairs(0)
    [[]]
    """
    # Input check
    if n < 0:
        raise ValueError(f'A negative number (got {n}) cannot be factorized')
    elif n == 0:
        return [[]]
    
    return [[i, n // i] for i in range(isqrt(n) + 1, 0, -1) if n % i == 0]

