from .ATask import ATask
import Initialization
from . import Functions


class Task(ATask):
    def __init__(self, signal_message, parameter, selections):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.parameter = parameter
        self.selections = selections
    
    def run(self):
        data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns=self.parameter, selections=self.selections)
        
        name_parameter = self.parameter.get_name()
        unique_values = data[name_parameter].value_counts(sort=False)
         
        unique_values = unique_values.astype('float')
        unique_values = self._calculate_per_one_day(unique_values)
        unique_values = self._normalize(unique_values)
        
        self.result = {}    
        self.result['data'] = unique_values
        self.result['size_selection'] = len(data)
    
    @staticmethod    
    def _calculate_per_one_day(unique_values):
        if unique_values.name == 'Birthmonth':
            func_count_days = Functions.get_quantity_days_in_month
        elif unique_values.name == 'ZodiacSign':
            func_count_days = Functions.get_quantity_days_in_zodiac
        else:
            return unique_values
        
        for index, value in unique_values.items():
            unique_values[index] = value / func_count_days(index)
        return unique_values
    
    @staticmethod
    def _normalize(unique_values):
        sum_values = sum(unique_values)
        if sum_values != 0:
            unique_values *= 100 / sum_values
        return unique_values
        
    
        