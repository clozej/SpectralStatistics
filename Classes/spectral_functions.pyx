from libc.math cimport sin, cos, fabs
cimport cython
from cython.parallel cimport prange

import numpy as np
cdef double PI = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_sff_parallel(double[:] E, double[:] tim,
                               double[:] weights, double[:] out) nogil:
    cdef int i, j
    cdef double w
    cdef double sff_real
    cdef double sff_imag
 
    for i in prange(tim.shape[0]):
        sff_imag = 0.0
        sff_real = 0.0
        for j in range(E.shape[0]):
            sff_real = sff_real + cos(E[j] * tim[i]* 2*PI) * weights[j]
            sff_imag = sff_imag + sin(E[j] * tim[i]* 2*PI) * weights[j]
        out[i] = sff_real**2 + sff_imag**2

def compute_sff(E, tim , weights):
    out = np.empty_like(tim, dtype='double')
    compute_sff_parallel( E, tim, weights, out)
    return tim, out
