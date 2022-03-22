import math as m
import numpy as np
from scipy import special

import sys
sys.path.append("..")

from Classes import Utils as ut

class Model:
    def __init__(self):
        self.par_bounds = None

    def P(self, s, n = 1):
        beta = 1 #using brody distribution with beta =1 gives Wigner-Dyson
        a = m.gamma((beta + 2) / (beta + 1)) ** (beta + 1)
        return (beta + 1) * a * np.power(s, beta) * np.exp(-a * np.power(s, beta + 1))

    def W(self, s, n = 1):
        beta = 1 #using brody distribution with beta =1 gives Wigner-Dyson
        a = m.gamma((beta + 2) / (beta + 1)) ** (beta + 1)
        return 1 - np.exp(-a * np.power(s, beta + 1))

    def U(self, s):
        return ut.distU(self.W( s))
    
    def NV(self,l):
        pl = np.pi*l
        pl2 = 2*np.pi*l 
        si1, ci1 = special.sici(pl) 
        si2, ci2 = special.sici(pl2) 
        
        return 2/np.pi**2*(np.log(pl2) + np.euler_gamma + 1 +0.5*si1**2 - np.pi/2*si1 -np.cos(pl2) - ci2 + np.pi**2*l*(1-2/np.pi*si2))

    def SFF(self, t):
        sff = 2*t - t*np.log(1+2*t)
        idx = t>1
        sff[idx] = 2 - t[idx]*np.log((2*t[idx]+1)/(2*t[idx]-1))
        return sff