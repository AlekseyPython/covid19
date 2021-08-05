import pandas as pd
from Entires.TypeMessage import TypeMessage
from Entires import ConvertedColumns, Functions
import Settings, Initialization


class ConvertedDataController:
    def __init__(self, signal_message, columns, selections, ignore_empty_values):
        if isinstance(columns, ConvertedColumns._Column):
            columns = [columns]
        elif columns is None:
            columns = Functions.get_columns(converted=True)
            
        self.signal_message = signal_message
        self.columns = columns
        self.selections = selections
        self.ignore_empty_values = ignore_empty_values
    
    def get_data(self):
        all_needed_columns = list(set(self.columns) | set(self.selections.get_columns()))
    
        file_name = Settings.CONVERTED_DATA
        data = Initialization.idata_source.read_hdf5_data(file_name, self.signal_message, columns=all_needed_columns)
        data = self._ignore_empty_values(data)
        data = self._select_values(data)
        data = self._delete_selections_columns(data)
        return data
                
    def _ignore_empty_values(self, data):
        if not self.ignore_empty_values:
            return data
        
        for column in self.columns:
            column_name = column.get_name()
            column_type = column.get_type()
            if column_type in ['bool', 'int8']:
                data = data[data[column_name] != -1]
            else:
                data = data[data[column_name].notnull()]
        return data
    
    def _select_values(self, data):
        for selection in self.selections:
            column = selection.get_column()
            compare_operation = selection.get_compare_operation()
            value = selection.get_value()
            
            column_name = column.get_name() 
            column_type = column.get_type()
            if column_type == 'bool':
                empty_value = value==Settings.EMPTY_BOOL
            elif column_type == 'int8':
                empty_value = value==Settings.EMPTY_INT
            elif column_type == 'datetime64[D]':
                empty_value = value==Settings.EMPTY_DATE
            else:
                empty_value = value is pd.NA
               
            if empty_value:
                if compare_operation == '=':
                    data = data[data[column_name].isnull()]
                elif compare_operation == '!=':
                    data = data[data[column_name].notnull()]
                else:
                    data = {}
                    data['type_message'] = TypeMessage.critical
                    data['text'] = 'For NA- values, only compare operations = and != can be used!'
                    self.signal_message.emit(data)
            else:
                if compare_operation == '=':   
                    data = data[data[column_name] == value]
                elif compare_operation == '!=':   
                    data = data[data[column_name] != value]
                elif compare_operation == '<':   
                    data = data[data[column_name] < value]
                elif compare_operation == '<=':   
                    data = data[data[column_name] <= value]
                elif compare_operation == '>':   
                    data = data[data[column_name] > value]
                elif compare_operation == '>=':   
                    data = data[data[column_name] >= value]
                else:
                    raise RuntimeError('An unknown type of operation was used in selections: ' + compare_operation)
        return data
        
    def _delete_selections_columns(self, data):
        columns_for_deleting = []
        for selection in self.selections:
            column = selection.get_column()
            if column in self.columns:
                continue
            
            columns_for_deleting.append(column.get_name())
        
        if columns_for_deleting:
            data = data.drop(set(columns_for_deleting), axis=1)   
        return data
        


