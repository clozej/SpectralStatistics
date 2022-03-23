import numpy as np
import sys
sys.path.append(".")
import Classes.Utils as ut
from spectral_functions import compute_sff

#define cumulative density function
def cdf(z):
    zz = np.sort(z)
    n = 1 + zz.size
    f = lambda x: np.count_nonzero(zz <= x) / n
    return np.vectorize(f)

class spectrum:

    def __init__(self, energy):
        self.energy = energy

    def spacing_ratio(self, shift = 1, n = 1):
        """Computes level spacing ratios of the spectra."""
        s = self.energy[n:] - self.energy[:-n]
        r = s[ : -shift ]/np.roll(s, -shift)[:-shift]
        return r

    def P(self, n=1, r_min=0, r_max = 4, grid = 200, log_x=False):
        """Computes level spacing ratio probability density function of the spectra."""
        s = self.spacing_ratio(n=n)
        h, bins = np.histogram(s, bins=grid+1, range=(s_min,s_max), density=True)
        return  (bins[1:] + bins[:-1])/2, h

    def W(self, n=1, r_min=0, r_max = 4, grid = 200, log_x=False):
        #"""Computes level spacing cumulative density function of unfolded spectra."""
        z = self.spacing_ratio(n=n) #/ np.mean(s)
        Wz = cdf(z)
        x = ut.sample(s_min, s_max, grid, log_x)
        y = Wz(x)
        return x, y
