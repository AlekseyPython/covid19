from .AController import AController
from Entires import Functions
from Entires.Enums import PeriodsPollution
from Entires.AirPollution import AirPollution
from Entires.TypeMessage import TypeMessage
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_columns_for_selection(self): 
        return Functions.get_columns(converted=True, prefixes_exclude=['Birthday'])
    
    def get_possible_values_parameters(self):
        parameters = {}
        parameters['Period pollution'] = [PeriodsPollution.last_month.value, PeriodsPollution.last_year.value]
        
        parameters_air_pollution = AirPollution.get_parameters_for_citizens(self.signal_message, PeriodsPollution.last_year.value) #get all parameters
        parameters['Parameter air pollution'] = parameters_air_pollution
        return parameters 
    
    def set_parameters(self, parameters):
        self.period_pollution = parameters['Period pollution']
        self.parameter_air_pollution = parameters['Parameter air pollution']
        
    def set_selections(self, selections):
        self.selections = selections
    
    def perform_task(self):
        return Initialization.ibusiness.get_air_pollutions_for_moscow_citizens(self.signal_message, self.period_pollution, self.parameter_air_pollution, self.selections)
        
    def show_result(self, result):
        if type(result['pollution_for_sick_peoples']) == str:
            Initialization.ipresentation.message_to_user(TypeMessage.information, text=result['pollution_for_sick_peoples'])
            return
        
        data = result['pollution_for_sick_peoples']
        data = data.rename(columns={'Longitude':'X', 'Latitude':'Y', self.parameter_air_pollution: 'Z'})
        
        parameters = {}
        parameters['title'] = 'Air pollution for Moscow citizens and data of meteo- stations located near highways.\nParameter = ' + self.parameter_air_pollution
        parameters['data_selections'] = self.selections
        parameters['size_selection'] = len(data)
        
        box_aspect = (1, 1.7, 1)
        fig = Initialization.ipresentation.plot_trisurf(data, box_aspect, **parameters)
        
        if self.parameter_air_pollution != 'AveragePDKs':
            data = result['pollution_near_road']
            data = data.rename(columns={'Longitude':'X', 'Latitude':'Y', 'AverageValue': 'Z'})
            fig = Initialization.ipresentation.scatter_3D(data, fig=fig)
        return fig

