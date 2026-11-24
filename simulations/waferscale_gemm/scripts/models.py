class ASIC:
    """ Model of an output-stationary systolic array computing
    C (m*n) = A (m*k) @ B (k*n), with one PE per element of C. """

    def __init__(self, m:int, k:int, n:int):
        """
        :param m: Number of rows of A
        :param k: Number of columns of A and number of rows of B
        :param n: Number of columns of B
        """
        if min(m, k, n) <= 0:
            raise ValueError('m, k, n must be positive, '
                             f'got m={m}, k={k}, n={n}')
        self.m = m
        self.k = k
        self.n = n

    @property
    def latency(self) -> int:
        """ ASIC cycles from the first operands entering the array to the
        last element of C being complete. """
        # The farthest PE waits (m-1) + (n-1) cycles for its first operands,
        # then accumulates k products.
        return self.m + self.k + self.n - 2
