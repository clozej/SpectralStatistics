import math as m
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def varProsen(L, E):
    Ave1 = 0.0
    Ave2 = 0.0
    Ave3 = 0.0
    Ave4 = 0.0
    j = 0
    k = 0
    x = E[0]
    #print(x)
    nstates = len(E)
    while x < E[-(int(L)+100)]:
        #print(k)
        while E[k] < x+L:
            k = k+1
        #continue
        d1 = E[j] - x
        d2 = E[k] - (x+L)
        cn = k - j
        if (d1 < d2):
            x = E[j]
            s = d1
            j = j + 1
        else:
            x = E[k] - L
            s = d2
            k = k + 1
        Ave1 = Ave1 + s*cn
        Ave2 = Ave2 + s*cn**2
        Ave3 = Ave3 + s*cn**3
        Ave4 = Ave4 + s*cn**4
        #print(s)
    s = E[-(int(L)+100)] - E[0]
    Ave1 = Ave1/s
    Ave2 = Ave2/s
    Ave3 = Ave3/s
    Ave4 = Ave4/s
    AveNum = (Ave1)
    AveSig = (Ave2 - AveNum**2)
    VarSig = (Ave4 - 4.*Ave3*AveNum + 8.*Ave2*AveNum**2 - 4.*AveNum**4 - Ave2**2)
    return AveNum , AveSig, VarSig


def weight_function(E,E0=0,sigma=1):
    return np.sqrt(2/(sigma*np.sqrt(np.pi))*np.exp(-4/(sigma**2)*np.power((E - E0), 2.)))

def distU(ws):
    return (2 / m.pi) * np.arccos(np.sqrt(1 - ws))

import inspect

def filter_dict(func, kwarg_dict):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}

    return filtered_dict

def sample(x_min, x_max, grid, log_x):
    if log_x:
        x = np.logspace(x_min, x_max, grid)
    else:
        x = np.linspace(x_min, x_max, grid)
    return x    

def set_axis(log_x, log_y):
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")


