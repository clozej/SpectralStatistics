import math as m
import numpy as np
from scipy import special

import sys
sys.path.append("..")

from Classes import Utils as ut

#helper functions
# brody gap probability
# brody distribution
def brodyP(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return (beta + 1) * a * np.power(s, beta) * np.exp(-a * np.power(s, beta + 1))

# cumulative brody distribution
def brodyW(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return 1 - np.exp(-a * np.power(s, beta + 1))

# brody gap probability
def brodyE(beta, s):
    a = m.gamma((beta + 2)/(beta + 1)) ** (beta + 1)
    x = a * np.power(s, beta + 1)
    return special.gammaincc(1/(beta + 1), x)

class Model:
    def __init__(self):
        self.par_bounds = [[0,0],[1,1]]

    def P(self, s, rho, beta):
        a = (rho ** 2) * brodyE(beta, (1 - rho) * s) - 2 * rho * (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) + ((1 - rho) ** 2) * brodyP(beta, (1 - rho) * s) 
        return a * np.exp(-rho * s)

    def W(self, s, rho, beta):
        a = (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) - rho * brodyE(beta, (1 - rho) * s)
        return a * np.exp(-rho * s) + 1

    def U(self, s, rho, beta):
        return ut.distU(self.W( s, rho, beta))
    