import numpy as np
import pandas as pd
import Settings
from Entires.Enums import PeriodsPollution
from Entires.FilesChecker import FilesChecker

class AirPollution:
    def __init__(self, signal_message, period_pollution, parameter, excluded_zones=[]):
        self.signal_message = signal_message
        self.period_pollution = period_pollution
        self.parameter = parameter
        self.excluded_zones = excluded_zones
        
        needed_columns = ['StationName', 'AverageValue']
        self.pollutions = self.get_air_pollution(signal_message, period_pollution, parameter, needed_columns)
        
        needed_columns=['StationName', 'Longitude', 'Latitude']
        self.characteristics_air_stations = self.get_characteristics_air_stations(signal_message, needed_columns, excluded_zones)
        if self.pollutions is None or self.characteristics_air_stations is None:
            raise RuntimeError('For creating AirPollution- object, you must have files with air pollution and characteristics of meteorological station.')

    def get_pollution_and_coordinates(self):
        common = self.pollutions.merge(self.characteristics_air_stations, on='StationName')
        common.drop(['StationName'], axis=1)
        return common
    
    @staticmethod
    def get_characteristics_air_stations(signal_message, needed_columns=['StationName', 'Zone', 'Longitude', 'Latitude'], excluded_zones=[]):
        def get_dtype_and_converters():
            crop_quotes = (lambda s: s[1: -1])
            crop_quotes_cast_float = (lambda s: float(s[1: -1] if len(s)>2 else 0))
            
            dtype = []
            converters = {}
            if 'StationName' in needed_columns:
                dtype.append(('StationName','U50'))
                converters[1] = crop_quotes
                
            if 'Zone' in needed_columns or excluded_zones:
                dtype.append(('Zone','U50'))
                converters[4] = crop_quotes
                
            if 'Longitude' in needed_columns:
                dtype.append(('Longitude','float64'))
                converters[8] = crop_quotes_cast_float
                
            if 'Latitude' in needed_columns:
                dtype.append(('Latitude','float64'))
                converters[9] = crop_quotes_cast_float
                
            return dtype, converters
        
        file_name = Settings.PATH_BASE_AIR_STATIONS
        checker = FilesChecker(signal_message)
        if not checker.existence(file_name):
            return None
        
        dtype, converters = get_dtype_and_converters()    
        usecols = tuple([key for key in converters.keys()])
        stations = np.genfromtxt(file_name, dtype, delimiter=';', skip_header=1, skip_footer=1, converters=converters, usecols=usecols, encoding='cp1251')
        
        #if there is only one column, then an ordinary array is returned (not structured), without names
        names = [value[0] for value in dtype]
        stations = pd.DataFrame(stations, columns=names)
        
        if excluded_zones:
            stations = stations[~stations['Zone'].isin(excluded_zones)]
            if 'Zone' not in needed_columns:
                stations = stations.drop('Zone', axis=1)
            
        return stations.drop_duplicates()
                
    @staticmethod
    def _get_start_end_period(period_pollution): 
        if period_pollution == PeriodsPollution.last_year.value:
            end_date = Settings.LAST_FULL_MONTH + np.timedelta64(1, 'M')
            start_date = end_date - np.timedelta64(1, 'Y')
            
        elif period_pollution == PeriodsPollution.last_month.value:
            start_date = Settings.LAST_FULL_MONTH
            end_date = start_date + np.timedelta64(1, 'M')
            
        else:
            raise RuntimeError('Please, give valid period for measuring air- pollution!')
        
        return start_date, end_date

    @staticmethod    
    def get_air_pollution(signal_message, period_pollution, parameters=[], needed_columns=['StationName', 'Parameter', 'AverageValue']):
        air_pollution = AirPollution._read_pollution_data(signal_message, period_pollution, needed_columns, parameters)
        if 'AverageValue' not in needed_columns:
            air_pollution = pd.DataFrame(air_pollution, columns=needed_columns)
            return air_pollution.drop_duplicates()
        
        #change value for parameters without PDK
        parameters_with_PDK = AirPollution._get_parameters(air_pollution, only_contain_PDK=True)
        mask = np.isin(air_pollution['Parameter'], parameters_with_PDK)
        air_pollution['AverageValue'][mask] = air_pollution['AverageValuePDKs'][mask]

        all_columns = ['StationName', 'Parameter', 'AverageValue']
        air_pollution = pd.DataFrame(air_pollution, columns=all_columns)
        
        if period_pollution == PeriodsPollution.last_year.value:
            grouping_columns = ['StationName', 'Parameter']
            air_pollution = air_pollution.groupby(grouping_columns).mean()
            air_pollution = air_pollution.reset_index()
            
        if 'StationName' not in needed_columns:
            air_pollution = air_pollution.drop('StationName', axis=1)
            
        if 'Parameter' not in needed_columns:
            air_pollution = air_pollution.drop('Parameter', axis=1)
            
        return air_pollution

    @staticmethod
    def _read_pollution_data(signal_message, period_pollution, needed_columns, parameters=[]):
        def get_dtype_and_converters():
            crop_quotes = (lambda s: s[1: -1])
            crop_quotes_cast_float = (lambda s: float(s[1: -1] if len(s)>2 else 0))
            cast_date_time = (lambda s: np.datetime64(s[4:8] + '-' + s[1:3]))
            
            dtype = [('MonthMeasuring', 'datetime64[M]')]
            converters = {1: cast_date_time}
            if 'StationName' in needed_columns or 'AverageValue' in needed_columns:
                dtype.append(('StationName','U50'))
                converters[3] = crop_quotes
                
            if 'Parameter' in needed_columns or 'AverageValue' in needed_columns or parameters:
                dtype.append(('Parameter','U50'))
                converters[8] = crop_quotes
                
            if 'AverageValue' in needed_columns:
                dtype.extend([('AverageValue','float64'),('AverageValuePDKs','float64')])
                converters[9] = crop_quotes_cast_float
                converters[10] = crop_quotes_cast_float
            return dtype, converters
    
        file_name = Settings.PATH_BASE_AIR_POLLUTIONS
        checker = FilesChecker(signal_message)
        if not checker.existence(file_name):
            return None
        
        dtype, converters = get_dtype_and_converters()    
        usecols = tuple([key for key in converters.keys()])
        air_pollution = np.genfromtxt(file_name, dtype, delimiter=';', skip_header=1, skip_footer=1, converters=converters, usecols=usecols, encoding='cp1251')
        
        start_date, end_date = AirPollution._get_start_end_period(period_pollution)   
        air_pollution = air_pollution[(air_pollution['MonthMeasuring']>=start_date) & (air_pollution['MonthMeasuring']<end_date)]
        
        if parameters:
            mask = np.isin(air_pollution['Parameter'], parameters)
            air_pollution = air_pollution[mask]
        return air_pollution
    
    @staticmethod
    def _get_parameters(air_pollution, only_contain_PDK=False):
        all_parameters = np.unique(air_pollution['Parameter'])
        if not only_contain_PDK:
            return list(all_parameters)
         
        parameters = []    
        for current_parameter in all_parameters:
            data_by_parameter = air_pollution[air_pollution['Parameter']==current_parameter]
            data_by_parameter_zero = data_by_parameter[data_by_parameter['AverageValuePDKs']==0]
            
            if only_contain_PDK and len(data_by_parameter)!=len(data_by_parameter_zero):
                parameters.append(current_parameter)
        return parameters
                   
    @staticmethod    
    def get_working_stations(signal_message, period_pollution, parameter=None):
        working_stations = AirPollution.get_air_pollution(signal_message, period_pollution, parameter, needed_columns=['StationName'])
        return list(map(str, working_stations['StationName']))
    
    @staticmethod    
    def get_parameters_for_stations(signal_message, period_pollution, only_contain_PDK=False):
        needed_columns=['Parameter']
        if only_contain_PDK:
            needed_columns.append('AverageValue')
            
        air_pollution = AirPollution._read_pollution_data(signal_message, period_pollution, needed_columns)
        parameters_for_stations = AirPollution._get_parameters(air_pollution, only_contain_PDK)
        return list(map(str, parameters_for_stations))
    
    @staticmethod    
    def get_parameters_for_citizens(signal_message, period_pollution):
        parameters = AirPollution.get_parameters_for_stations(signal_message, period_pollution)
        parameters.append('AveragePDKs')
        return parameters
    
    @staticmethod
    def get_zones(signal_message):
        characteristics = AirPollution.get_characteristics_air_stations(signal_message, needed_columns=['Zone'])
        return list(map(str, characteristics['Zone']))
         
        
    