import pandas as pd
from .ATask import ATask
from Entires import Enums
from Entires.ConvertedColumns import Longitude, Latitude
from Entires.AirPollution import AirPollution
import Initialization


class Task(ATask):
    def __init__(self, signal_message, period_pollution, parameter_air_pollutin, selections):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.period_pollution = period_pollution
        self.parameter_air_pollutin = parameter_air_pollutin
        self.selections = selections
    
    def run(self):
        self.result = {} 
        
        columns=[self.parameter_air_pollutin]
        table_name = Enums.get_table_hdf5_by_period_pollution(self.period_pollution)
        air_pollution = Initialization.icontroller_data_sourse.read_prepared_data(self.signal_message, table_name, columns)
        if len(air_pollution.columns) == 0:
            self.result['pollution_for_sick_peoples'] = 'Nothing data for this parameter'
            return
        
        #get Longitude and Latitude
        columns = [Longitude, Latitude]
        converted_data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns, self.selections)
        self.result['pollution_for_sick_peoples'] = pd.merge(converted_data, air_pollution, left_index=True, right_index=True)

        #get data for road
        zones = AirPollution.get_zones(self.signal_message)
        excluded_zones = [zone for zone in zones if zone!='Вблизи автомагистралей']
        air_pollution = AirPollution(self.signal_message, self.period_pollution, self.parameter_air_pollutin, excluded_zones)
        self.result['pollution_near_road'] = air_pollution.get_pollution_and_coordinates()
        
