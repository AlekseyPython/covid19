import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator as CubicNDInterpolator
import Settings


class Interpolator2D:
    def __init__(self, air_pollution):
        self.air_pollution = air_pollution
        self.interpolators = self._get_interpolators()
    
    def get_inrepolations(self, X, Y):
        if self.interpolators is None:
            #few stations measure this pollution parameter 
            return None
        
        inrepolations = self.interpolators['linear'](X, Y)
        
        X_nearest = X[np.isnan(inrepolations)]
        Y_nearest = Y[np.isnan(inrepolations)]
        if len(X_nearest):
            inrepolations_nearest = self.interpolators['nearest'](X_nearest, Y_nearest)
            if inrepolations_nearest[np.isnan(inrepolations_nearest)]:
                raise RuntimeError('Interpolation by the nearest neighbor returned NA- values!')
            
        inrepolations[np.isnan(inrepolations)] = inrepolations_nearest
        return inrepolations
    
    def _get_interpolators(self):
        pollution_and_coordinates = self.air_pollution.get_pollution_and_coordinates()
        if len(pollution_and_coordinates) < Settings.MINIMUM_QUANTITY_STATIONS_FOR_CALCULATING_AIR_POLLUTION:
            return None
        
        longitudes = pollution_and_coordinates['Longitude']
        latitudes = pollution_and_coordinates['Latitude']
        domain = list(zip(longitudes, latitudes))
        
        pollution_values = pollution_and_coordinates['AverageValue']
        
        interpolators = {}
        interpolators['linear'] = LinearNDInterpolator(domain, pollution_values)#CubicNDInterpolator(domain, pollution_values)
        interpolators['nearest'] = NearestNDInterpolator(domain, pollution_values)
        return interpolators
        
