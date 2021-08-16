import numpy as np
from .AController import AController
from Entires import Functions, Enums, GeoRegion
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_columns_for_selection(self): 
        return Functions.get_columns(converted=True)
        
    def get_possible_values_parameters(self):
        return {'Type data': ['Spread of disease', 'Meteorological stations']}  
    
    def set_parameters(self, parameters):
        self.type_data = parameters['Type data']
        
    def set_selections(self, selections):
        self.selections = selections
    
    def perform_task(self):
        return Initialization.ibusiness.get_spread_disease(self.signal_message, self.type_data, self.selections)
        
    def show_result(self, result):
        name = Enums.MoscowAreas.old_moscow.value
        moscow_region = GeoRegion.GeoRegion(name, names_shape_files=[name])
        
        if self.type_data == 'Spread of disease':
            params = {}
            params['region'] = moscow_region
            params['background_data'] = result['leaving_houses'] 
            region_plotter = Initialization.ipresentation.create_region_plotter(**params)
            Initialization.ipresentation.show_region(region_plotter)
            
            sick_people = result['sick_people']
            sick_people['color'] = np.where(sick_people['Traveler'], 'red', 'green')
            
            params = {}
            params['data'] = sick_people
            params['column_slide_separator'] = 'DateCreating'
            params['title'] = 'Spread of disease for Moscow'
            params['region_plotter'] = region_plotter
            params['slide_delay'] = 0
            params['selections'] = self.selections
            params['save_pictures'] = False
            Initialization.ipresentation.slide_show_of_geo_data(**params)
            
        else:
            sick_people = result['sick_people']
            sick_people = sick_people.value_counts(['Longitude','Latitude'])
            sick_people = sick_people.reset_index()
            
            #remove column with count unique values 'Longitude' and 'Latitude'
            removable_column = [col for col in sick_people.columns if col not in ['Longitude','Latitude']]
            sick_people = sick_people.drop(removable_column, axis=1)
             
            params = {}
            params['region'] = moscow_region
            params['background_data'] = sick_people
            region_plotter = Initialization.ipresentation.create_region_plotter(**params)
            Initialization.ipresentation.show_region(region_plotter)
            
            stations = result['stations']
            stations['shape'] = 30
            
            params = {}
            params['data'] = stations
            params['title'] = 'Moscow meteorological stations'
            params['region_plotter'] = region_plotter
            params['selections'] = self.selections
            params['show_data_only_in_region'] = False
            Initialization.ipresentation.show_geo_data(**params)
        
        return region_plotter