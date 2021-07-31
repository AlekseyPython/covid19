from .ATask import ATask
from Entires import ConvertedColumns
from . import Functions
import Initialization


class Task(ATask):
    def __init__(self, signal_message, parameter, selections):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.parameter = parameter
        self.selections = selections
    
    def run(self):
        columns = [ConvertedColumns.DateCreating, self.parameter]
#         columns = [ConvertedColumns.DateLeavingFromHospital, self.parameter]
        
        data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns, self.selections)
        unique_values = data.value_counts(sort=False)
        
        unique_values = unique_values.astype('float')
        unique_values = self._calculate_per_one_day(unique_values)
        
        name_date = ConvertedColumns.DateCreating.get_name()
#         name_date = ConvertedColumns.DateLeavingFromHospital.get_name()
        unique_values = self._normalize(unique_values, name_date)
        
        list_of_names = unique_values.index.names
        if list_of_names.index(name_date) == 1:
            name_parameter = self.parameter.get_name()
            unique_values = unique_values.swaplevel(name_date, name_parameter)
        
        self.result = {}
        self.result['data'] = unique_values.unstack()
        self.result['size_selection'] = len(data)
    
    @staticmethod    
    def _calculate_per_one_day(unique_values):
        if 'Birthmonth' in unique_values.index.names:
            func_count_days = Functions.get_quantity_days_in_month
            position_parameter = unique_values.index.names.index('Birthmonth')
        elif 'ZodiacSign' in unique_values.index.names:
            func_count_days = Functions.get_quantity_days_in_zodiac
            position_parameter = unique_values.index.names.index('ZodiacSign')
        else:
            return unique_values
        
        for index, value in unique_values.items():
            value_parameter = index[position_parameter]
            unique_values[index] = value / func_count_days(value_parameter)
        return unique_values
    
    @staticmethod
    def _normalize(unique_values, name_date):
        position_date = unique_values.index.names.index(name_date)
        collection_for_search_by_date = unique_values.sum(level=name_date)
        collection_for_search_by_date.sort_index(inplace=True)
            
        for index, value in unique_values.items():
            value_date = index[position_date]
            unique_values[index] = 100 * value / collection_for_search_by_date[value_date]
        return unique_values
            
    
        