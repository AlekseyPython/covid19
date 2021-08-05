from .AController import AController
from Entires import Functions
from Entires.TypeMessage import TypeMessage
from Entires.Enums import FittingCurve
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_columns_for_selection(self): 
        return Functions.get_columns(converted=True, prefixes_exclude=['Birthday'])
    
    def get_possible_values_parameters(self):
        parameters = {}
        parameters['Analyzing column'] = Functions.get_columns(converted=True, prefixes_exclude=['Birthday'])
        parameters['Fitting curve'] = ['', FittingCurve.gompertz.value] 
        return parameters 
    
    def set_parameters(self, parameters):
        self.analyzing_column = parameters['Analyzing column']
        self.fitting_curve = parameters['Fitting curve']
        
    def set_selections(self, selections):
        if not selections:
            Initialization.ipresentation.message_to_user(TypeMessage.critical, text='The selection table cannot be empty!')
        self.selections = selections
    
    def perform_task(self):
        ibusiness = Initialization.ibusiness
        return ibusiness.get_probability_according_to_parameter(self.signal_message, self.analyzing_column, self.selections, self.fitting_curve)
        
    def show_result(self, result):
        result['ylabel'] = '%'
        result['title'] = self.analyzing_column.get_name()
        result['data_selections'] = self.selections
        
        optimal_parameters = result['optimal_parameters']
        result.pop('optimal_parameters')
        
        R2 = result['R2']
        result.pop('R2')
        
        if self.analyzing_column.get_name() == 'Age':
            fig = Initialization.ipresentation.plot(**result)
        else:
            fig = Initialization.ipresentation.build_bar(**result)
            
        if not self.fitting_curve:
            return fig
        
        result = {}
        result['fitting_curve'] = self.fitting_curve
        result['optimal_parameters'] = optimal_parameters
        result['R2'] = R2
        result['fig'] = fig
        return Initialization.ipresentation.plot_curve(**result)
            
        
        
            