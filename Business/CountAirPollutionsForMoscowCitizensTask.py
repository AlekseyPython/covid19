import numpy as np
import pandas as pd
from Entires import Enums
from Entires.AirPollution import AirPollution
from Entires.Interpolator2D import Interpolator2D
from Entires.ConvertedColumns import Longitude, Latitude
import Initialization
from .ATask import ATask


class Task(ATask):
    def __init__(self, signal_message, include_stations_near_the_roads):
        ATask.__init__(self)
        self.signal_message = signal_message
        self.include_stations_near_the_roads = include_stations_near_the_roads
    
    def run(self):
        self.result = {}
        self.result['data'] = self._count()
    
    def _count(self):
        converted_data = self._get_converted_data()
        for period_enum in Enums.PeriodsPollution:
            period_pollution = period_enum.value
            air_polutions = self._get_air_polutions_for_period(converted_data, period_pollution)
            
            table_name = Enums.get_table_hdf5_by_period_pollution(period_enum)
            Initialization.icontroller_data_sourse.write_prepared_data(self.signal_message, air_polutions, table_name)
        return True
    
    def _get_air_polutions_for_period(self, converted_data, period_pollution):    
        excluded_zones = []
        if not self.include_stations_near_the_roads:
            excluded_zones.append('Вблизи автомагистралей')
            
        longitude = converted_data['Longitude'].to_numpy()
        latitude = converted_data['Latitude'].to_numpy()
        
        all_pollutions = {} 
        parameters_for_stations = AirPollution.get_parameters_for_stations(self.signal_message, period_pollution, only_contain_PDK=False)
        for parameter in parameters_for_stations:
            air_pollution = AirPollution(self.signal_message, period_pollution, parameter, excluded_zones)
            interpolator = Interpolator2D(air_pollution)
            predicted_pollution = interpolator.get_inrepolations(longitude, latitude)
            if predicted_pollution is None:
                #few stations measure this pollution parameter
                continue
            
            all_pollutions[parameter] = pd.Series(predicted_pollution, name=parameter, dtype='float64')
        
        all_pollutions = pd.DataFrame(all_pollutions)
        all_pollutions = self._fill_average_pdk(all_pollutions, period_pollution)
        all_pollutions = self._set_index_from_converted_data(all_pollutions, converted_data)
        return all_pollutions
    
    def _fill_average_pdk(self, all_pollutions, period_pollution):
        parameters_with_PDK = AirPollution.get_parameters_for_stations(self.signal_message, period_pollution, only_contain_PDK=True)
        
        parameters_with_PDK = np.array(parameters_with_PDK)
        mask = np.isin(parameters_with_PDK, all_pollutions.columns)
        parameters_with_PDK = parameters_with_PDK[mask]
        
        all_pollutions['AveragePDKs'] = all_pollutions[parameters_with_PDK].mean(axis=1)
        return all_pollutions
    
    def _set_index_from_converted_data(self, df, converted_data):
        return df.set_index(converted_data.index)
        
    def _get_converted_data(self):
        columns = [Longitude, Latitude]
        return Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns=columns)
        
        
        