from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import curve_fit


class _Fitter(metaclass=ABCMeta):
    def __init__(self):pass
        
    def fit(self, x, y):
        func = self._fitting_function
        p0 = self._get_p0()
        sigma = self._get_sigma()
        bounds = self._get_bounds()
        popt, _ = curve_fit(func, x, y, p0, sigma, bounds=bounds)
        R2 = self._calculate_R2(func, x, y, sigma, popt)
        return popt, R2
    
    @staticmethod
    @abstractmethod
    def _fitting_function(): pass
    
    @staticmethod
    def _get_p0(): 
        return None
    
    @abstractmethod    
    def _get_sigma(self):
        return None
    
    @staticmethod
    def _get_bounds():
        return (0, np.inf)
    
    @staticmethod
    def _calculate_R2(func, x, y, sigma, popt):
        if sigma is None:
            sigma = np.ones_like(y)
            
        y_calc = func(x, *popt)
        d1 = (y-y_calc) / sigma
        d2 = (y-y.mean()) / sigma
        R2 = 1 - d1.dot(d1) / d2.dot(d2)
        return R2


class GompertzFitter(_Fitter):
    def __init__(self, sizes_selections=None):
        _Fitter.__init__(self)
        
        self.sizes_selections = sizes_selections
        self.probabilities = None
    
    def fit(self, x, y):
        for _ in range(2):
            popt, _ = _Fitter.fit(self, x, y)
            self.probabilities = self._fitting_function(x, *popt)
            
        return _Fitter.fit(self, x, y)
        
    @staticmethod
    def _fitting_function(x, a, b, c):
        return a * np.exp(b*x) + c
    
    @staticmethod    
    def _get_p0():
        return [0.0002, 0.07, 0]
    
    def _get_sigma(self):
        probabilities = self.probabilities
        sizes_selections = self.sizes_selections
        if probabilities is None and sizes_selections is None:
            return None
        
        elif probabilities is None:
            return np.sqrt(1/sizes_selections)
        
        elif sizes_selections is None:
            return np.sqrt(probabilities * (1-probabilities))
        
        else: 
            return np.sqrt(probabilities * (1-probabilities) / sizes_selections)
        
    @staticmethod
    def _get_bounds():
        return (0., [1., 1., 1.])

