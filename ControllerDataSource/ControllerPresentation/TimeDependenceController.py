from .AController import AController
from Entires import Functions
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_columns_for_selection(self): 
        return Functions.get_columns(converted=True, prefixes_exclude=['Birthday'])
        
    def get_possible_values_parameters(self):
        parameters = {}
        parameters['Analyzing column'] = Functions.get_columns(converted=True, prefixes_exclude=['Birthday', 'Age', 'DateCreating', 'DateAnalysis'])
        return parameters  
    
    def set_parameters(self, parameters):
        self.analyzing_column = parameters['Analyzing column']
        
    def set_selections(self, selections):
        self.selections = selections
    
    def perform_task(self):
        return Initialization.ibusiness.get_time_dependence(self.signal_message, self.analyzing_column, self.selections)
        
    def show_result(self, result):
        result['ylabel'] = '%'
        result['title'] = self.analyzing_column.get_name()
        result['data_selections'] = self.selections
        return Initialization.ipresentation.plot(**result)