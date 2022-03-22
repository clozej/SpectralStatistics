import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import sys
sys.path.append("..")

from scipy.optimize import curve_fit
from inspect import signature
import inspect
from Classes import Utils as ut
#from Models import GOE

class spectral_statistics:

    def __init__(self, data_sets, models):
        if not isinstance(data_sets, list):
            self.data_sets = [data_sets]
        else:
            self.data_sets = data_sets

        if not isinstance(models, list):
            self.models = [models]
        else:
            self.models = models 

    def plot_data(self, statistic, idx = 0, log_x=False, log_y=False, 
                    plot_kwargs={}, **kwargs):
        data = self.data_sets[idx]
        fun = getattr(data, statistic)
        fun_kwargs =ut.filter_dict(fun, kwargs)
        #print(fun_kwargs)
        x, y = fun(**fun_kwargs)
        #print(plot_kwargs)
        plt.plot(x,y,**plot_kwargs)
        ut.set_axis(log_x, log_y)

    def plot_model(self, statistic, *args, idx = 0, x_min=0, x_max=5, grid = 200,
                    log_x=False, log_y=False,  plot_kwargs={}, **kwargs):
        model = self.models[idx]
        fun = getattr(model, statistic)
        x = ut.sample(x_min, x_max, grid, log_x)
        y = fun(x,*args, **kwargs)
        plt.plot(x,y,**plot_kwargs)
        ut.set_axis(log_x, log_y)


    def fit(self, statistic, idx1 = 0, idx2= 0, plot_fit=True, 
                log_x=False, log_y=False, plot_kwargs_data ={},
                plot_kwargs_model ={}, **kwargs):

        data = self.data_sets[idx1]
        model = self.models[idx2]
        bounds = model.par_bounds

        fun = getattr(data, statistic)
        x, y = fun(**kwargs)

        fit_fun = getattr(model, statistic)
        #sig = signature(fit_fun)
        #pars = [i.default is inspect.Parameter.empty for i in list(sig.parameters.values())]
        #n = sum(pars)
        #print(sum(pars))
        #print([i.default for i in sig.parameters ])
        params = None
        if bounds is not None:         
            popt, pcov = curve_fit(fit_fun, x, y, bounds = bounds)
            params = popt
        print(params)
        if plot_fit:
            plt.plot(x,y,**plot_kwargs_data)
            x_fit = x
            if bounds is not None: 
                y_fit = fit_fun(x_fit, *popt)
            else:
                y_fit = fit_fun(x_fit)
            plt.plot(x,y_fit,**plot_kwargs_model)
            ut.set_axis(log_x, log_y)
        return params

    def plot_difference(self, statistic, idx1 = 0, idx2= 0, 
                log_x=False, log_y=False, 
                plot_kwargs ={}, **kwargs):

        data = self.data_sets[idx1]
        model = self.models[idx2]
        bounds = model.par_bounds

        fun = getattr(data, statistic)
        x, y = fun(**kwargs)

        fit_fun = getattr(model, statistic)
        #sig = signature(fit_fun)
        #pars = [i.default is inspect.Parameter.empty for i in list(sig.parameters.values())]
        #n = sum(pars)
        #print(sum(pars))
        #print([i.default for i in sig.parameters ])
        params = None
        if bounds is not None:         
            popt, pcov = curve_fit(fit_fun, x, y, bounds = bounds)
            params = popt
        print(params)
        x_fit = x
        if bounds is not None: 
            y_fit = fit_fun(x_fit, *popt)
        else:
            y_fit = fit_fun(x_fit)
        plt.plot(x,y-y_fit,**plot_kwargs)
        ut.set_axis(log_x, log_y)
        return params
