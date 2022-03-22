import math as m
import numpy as np
from scipy import special

import sys
sys.path.append("..")

from Classes import Utils as ut

class Model:
    def __init__(self):
        self.par_bounds = [0,np.inf]
        
    def P(self, s, a, n =1):
        b = a/n
        return b**a/special.gamma(a)*s**(a-1)*np.exp(-b*s)

    def W(self, s, a, n = 1):
        b = a/n
        return special.gammainc(a,b*s)#/sc.gamma(a)


    def U(self, s, a, n = 1):
        return ut.distU(self.W( s,a, n = n))