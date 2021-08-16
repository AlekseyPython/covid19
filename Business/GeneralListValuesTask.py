import numpy as np
from functools import reduce
from fuzzywuzzy import  process
from . import Functions
from .ATask import ATask
from Entires import SourceColumns
import Initialization, Settings


class Task(ATask):
    def __init__(self, signal_message, parameter, source_of_data_is_converted):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.parameter = parameter
        self.source_of_data_is_converted = source_of_data_is_converted
        
    def run(self):
        if self.source_of_data_is_converted:
            data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, columns=self.parameter, ignore_empty_values=False)
            
            for column in self.parameter:
                name = column.get_name()
                if column.numeric_type():
                    data[data[name] == Settings.EMPTY_INT] = 'NA'
                
                elif column.date_type():
                    data[data[name] == Settings.EMPTY_DATE] = 'NA'
                    
                elif column.bool_type():
                    data[data[name] == Settings.EMPTY_BOOL] = 'NA'
                
            unique_values = Functions.get_counted_unique_values_dataframe(data)
        else:
            data = Initialization.icontroller_data_sourse.read_source_data(self.signal_message, columns=self.parameter)
            
            for column in self.parameter:
                if column == SourceColumns.ListOfMedicines:
                    name_column = column.get_name()
                    data = self._get_each_word_of_data(data, name_column)
                    data = self._replace_misspelled_words(data, name_column)
                
            unique_values = Functions.get_counted_unique_values_nd_array(data)
            
        unique_values = Functions.sort_dict_by_values(unique_values, forward_direction=False)
        
        self.result = {}
        self.result['data'] = unique_values
    
    def _get_each_word_of_data(self, data, name_column):
        words = []
        table_of_simbols = {}
        table_of_simbols[ord(',')] = ord(' ')
        table_of_simbols[ord('.')] = ord(' ')
        table_of_simbols[ord(':')] = ord(' ')
        table_of_simbols[ord(';')] = ord(' ')
        table_of_simbols[ord('-')] = ord(' ')
        table_of_simbols[ord('+')] = ord(' ')
        table_of_simbols[ord('/')] = ord(' ')
        table_of_simbols[ord('|')] = ord(' ')
        table_of_simbols[ord('\\')] = ord(' ')
        table_of_simbols[ord('\t')] = ord(' ')
        
        for string_data in data:
            all_string = string_data[name_column]
            if not all_string:
                continue
            
            all_string = all_string.lower()
            all_string = all_string.translate(table_of_simbols)
            words_of_string = all_string.split()
            for word in words_of_string:
                if len(word) <= 5:
                    continue
                
                if word in Settings.REPLACEMENTS_LIST_OF_MEDICINES:
                    continue
                
                words.append(word)
            
        structured_array = np.array(words, dtype=[(name_column, 'U20')])
        return structured_array
    
    def _replace_misspelled_words(self, data, name_column):
        unique_values = Functions.get_counted_unique_values_nd_array(data)
        unique_values = Functions.sort_dict_by_values(unique_values, forward_direction=False)
        
        dictionary = []
        
        share = 0
        summa = reduce(lambda x,y: x+y, unique_values.values())
        for word, value in unique_values.items():
            if dictionary:
                close_word, distance = process.extractOne(word, dictionary)
                if distance > 70:
                    continue
                
            dictionary.append(word)
            share += value / summa
            if share >= 0.96:
                break
        
        replacement_words = []
        for row in data:
            word = row[name_column]
            close_word, distance = process.extractOne(word, dictionary)
            if distance > 75:
                replacement_words.append(close_word)
            else:
                replacement_words.append(word)
                
        structured_array = np.array(replacement_words, dtype=[(name_column, 'U20')])
        return structured_array
                
        
                
                
                
            
            
            
        
        
                       
        