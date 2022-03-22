import math as m
import numpy as np
from scipy import special

import sys
sys.path.append("..")

from Classes import Utils as ut

class Model:
    def __init__(self):
        self.par_bounds = [0,1]
        
    def P(self, s, beta):
        a = m.gamma((beta + 2) / (beta + 1)) ** (beta + 1)
        return (beta + 1) * a * np.power(s, beta) * np.exp(-a * np.power(s, beta + 1))

    def W(self, s, beta):
        a = m.gamma((beta + 2) / (beta + 1)) ** (beta + 1)
        return 1 - np.exp(-a * np.power(s, beta + 1))

    def U(self, s, beta):
        return ut.distU(self.W( s, beta))
    