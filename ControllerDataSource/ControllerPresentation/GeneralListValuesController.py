from functools import reduce
from .AController import AController
from Entires import SourceColumns, ConvertedColumns, SimilarColumns, Functions
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_possible_values_parameters(self):
        if self.source_of_data_is_converted:
            module = ConvertedColumns
            converted = True
        else:
            module = SourceColumns
            converted = False
        
        columns = [module.all_result]   
        columns.extend(Functions.get_columns(converted, prefixes_exclude=['Comment'])) 
        return {'Analyzing column': columns}    
    
    def set_parameters(self, parameters):
        self.analyzing_column = parameters['Analyzing column']
        
    def get_radio_buttons(self):
        return ['Source csv- data', 'Converted hdf5- data']
    
    def set_radio_button(self, values):
        self.source_of_data_is_converted = values['Converted hdf5- data']
        
    def perform_task(self):
        if type(self.analyzing_column) == SimilarColumns.SimilarColumns:
            columns = self.analyzing_column.get_columns()
        else:
            columns = [self.analyzing_column]
            
        return Initialization.ibusiness.get_general_list_values(self.signal_message, columns, self.source_of_data_is_converted)
    
    def show_result(self, result):
        text = ''
        data = result['data']
        summa = reduce(lambda x,y: x+y, data.values())
        for key, value in data.items():
            persent = round(100 * value / summa, 1)
            text += str(key) + ' = ' + str(value) + ' (' + str(persent) + '%)\n'
        
        title = self.analyzing_column.get_name()        
        return Initialization.ipresentation.show_text_in_widget(text, title)
    