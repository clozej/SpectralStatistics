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
        return np.exp(-s)

    def W(self, s, n = 1):
        return 1-np.exp(-s)

    def U(self, s):
        return ut.distU(self.W( s))
    
    def NV(self,l):
        return l

    def SFF(self, t):
        sff = np.ones(len(t))
        return sff