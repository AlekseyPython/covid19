import numpy as np
import pandas as pd
from .ATask import ATask
from Entires import ConvertedColumns as CC
from Entires.AirPollution import AirPollution
from Entires.Enums import PeriodsPollution
import Initialization, Settings


class Task(ATask):
    def __init__(self, signal_message, type_data, selections):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.type_data = type_data
        self.selections = selections
    
    def run(self):
        self.result = {} 
        
        columns = [CC.DateCreating, CC.Country, CC.Longitude, CC.Latitude]
        sick_people = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns, self.selections, ignore_empty_values=False)
        sick_people = sick_people.dropna(subset=['Latitude', 'Longitude'])
                           
        sick_people['Traveler'] = -sick_people['Country'].isna()
        sick_people = sick_people.drop(columns=['Country'])
        self.result['sick_people'] = sick_people
            
        if self.type_data == 'Spread of disease':
            file_name = Settings.PATH_BASE_TYPE_BUILDING_AND_COORDINATES
            dtype = [('Building','U50'),('Latitude','float64'),('Longitude','float64')]
            
            all_buildings = np.genfromtxt(file_name, dtype, delimiter=';', skip_header=1, skip_footer=1)
            all_leaving_houses = all_buildings[(all_buildings['Building'] == 'apartments') | (all_buildings['Building'] == 'residential')]
            pd_all_leaving_houses = pd.DataFrame(all_leaving_houses, columns=['Latitude', 'Longitude'])
            self.result['leaving_houses'] = pd_all_leaving_houses
            
        elif self.type_data == 'Meteorological stations':
            stations = AirPollution.get_characteristics_air_stations(self.signal_message, excluded_zones=['Вблизи автомагистралей'])
            
            period_pollution = PeriodsPollution.last_year.value
            working_stations = AirPollution.get_working_stations(self.signal_message, period_pollution)
            
            self.result['stations']  = stations[stations['StationName'].isin(working_stations)]    
            
        else:
            raise RuntimeError('Unknown parameter value!')