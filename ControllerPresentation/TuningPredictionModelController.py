from .AController import AController
import Initialization
from Entires.Enums import AnalysisTypes, OperationsOfMachineLearning, get_values_enum, ResultOperationOfMachineLearning
from Entires.TypeMessage import TypeMessage


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_possible_values_parameters(self):
        parameters = {}
        parameters['Type of analyzing'] = get_values_enum(AnalysisTypes)
        parameters['Operation']         = get_values_enum(OperationsOfMachineLearning)
        parameters['Severity of disease (empty for all)'] = ['', 'Средней тяжести', 'Тяжелое течение', 'Умер']
        return parameters
    
    def get_radio_buttons(self):
        return ['Print only final results', 'Print all messages']
    
    def set_radio_button(self, values):
        self.print_only_final_results = values['Print only final results']
        
    def set_parameters(self, parameters):
        self.analysis_type  = parameters['Type of analyzing']
        self.operation      = parameters['Operation']
        
        severity_of_disease = parameters['Severity of disease (empty for all)']
        if severity_of_disease:
            self.severity_of_disease = [severity_of_disease]
        else:
            possible_values_parameters = self.get_possible_values_parameters()
            self.severity_of_disease = possible_values_parameters['Severity of disease (empty for all)'][1:]
            
    def perform_task(self):
        return Initialization.ibusiness.tuning_prediction_model(self.signal_message, self.analysis_type, self.operation, self.severity_of_disease, self.print_only_final_results)
    
    def show_result(self, result):
        def representation_dict(dic):
            common_text = ''
            for key, value in dic.items():
                common_text += str(key) + ' = ' + str(value) + '\n'
            return common_text
        
        if result is None:
            return
        
        common_text = ''
        for element in result:
            if type(element) == ResultOperationOfMachineLearning:
                common_text += 'Operation = ' + element.operation + '\n'
                common_text += representation_dict(element.parameters)
                common_text += representation_dict(element.metrics)

            elif type(element) == dict:
                common_text += representation_dict(element)
                
            else:
                common_text += str(element) + '\n'
            common_text += '\n'
        
        common_text = common_text.strip()    
        if not common_text or common_text=='None':
            text = 'Tuning a prediction model for disease severity'
            informative_text = 'Operation successfully completed!'
            return Initialization.ipresentation.message_to_user(TypeMessage.information, text, informative_text)
        
        title = self.operation
        return Initialization.ipresentation.show_text_in_widget(common_text, title)
                
    
    