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

    def spacing(self, n = 1):
        """Computes n-th nearest neighbour level spacing of unfolded spectra."""
        s = self.energy[n:] - self.energy[:-n]
        return s

    def P(self, n=1, s_min=0, s_max = 4, grid = 200, log_x=False):
        """Computes level spacing probability density function of unfolded spectra."""
        s = self.spacing(n=n)
        h, bins = np.histogram(s, bins=grid+1, range=(s_min,s_max), density=True)
        return  (bins[1:] + bins[:-1])/2, h

    def W(self, n=1, s_min=0, s_max = 4, grid = 200, log_x=False):
        """Computes level spacing cumulative density function of unfolded spectra."""
        z = self.spacing(n=n) #/ np.mean(s)
        Wz = cdf(z)
        x = ut.sample(s_min, s_max, grid, log_x)
        y = Wz(x)
        return x, y

    def U(self, s_min=0, s_max = 4, grid = 200, log_x=False):
        """Computes transformed level spacing cumulative density function of unfolded spectra."""
                
        x, ws = self.W(n=1, s_min=s_min, s_max = s_max, grid = grid, log_x=log_x)
        u = ut.distU(ws)
        return x, u

    
    def NV(self, l_min=0, l_max = 20, grid = 50, log_x=False):
        """Computes number variance of unfolded spectra."""
        E = self.energy
        Ls = ut.sample(l_min, l_max, grid, log_x)
        data = [ut.varProsen(L, E) for L in Ls]
        num, sig, varsig =  np.array(data).transpose()
        return Ls, sig

    def SFF(self, a=1, min_t = 0, max_t = 3, weight_function = None, **kwargs):
        """Computes spectral form factor of unfolded spectra in time units of Heisenberg time."""
        E = self.energy
        E0 = np.mean(E)
        W = E[-1]-E[0]
        dt = round(a/(W),7)
        print("dt = %s" %dt)
        print("E0 = %s" %E0)
        print("W = %s" %W)
        tim = np.arange(min_t ,max_t, dt)
            
        if weight_function is not None:
            weights = weight_function(E, **kwargs)
        else:
            weights = np.ones(len(E))
        tim, sff = compute_sff(E,tim,weights)
        return tim, sff
