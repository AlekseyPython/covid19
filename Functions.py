import os
import json
from datetime import datetime
import numpy as np
import Settings


def save_object_to_file(obj, parameters_temporary_file):
    dir_name = Settings.FOLDER_FOR_TEMPORARY_FILES
    file_name = _get_temporary_file_name(parameters_temporary_file, with_date_and_time=True)
    full_path = dir_name + '/' + file_name
    with open(full_path, 'w') as f:
        json.dump(obj, f)

def get_temporary_files(parameters_temporary_file):
    end_of_file_name = _get_temporary_file_name(parameters_temporary_file, with_date_and_time=False)
    dir_name = Settings.FOLDER_FOR_TEMPORARY_FILES
    return [file_name for file_name in os.listdir(dir_name) if file_name.endswith(end_of_file_name)]

def count_quantity_files(parameters_temporary_file):    
    file_names = get_temporary_files(parameters_temporary_file)
    return len(file_names)
    
def load_objects_from_complete_bunch_of_files(parameters_temporary_files: list):
    files_for_parameters = {}
    for parameters in parameters_temporary_files:
        files_for_parameters[parameters] = sorted(get_temporary_files(parameters), reverse=True)
    
    #remove from lists all unordered files    
    odjects = {}
    for parameters, files in reversed(files_for_parameters.items()):
        if not files:
            return odjects
        
        last_file = files[0] 
        for sub_parameters, sub_files in files_for_parameters.items():
            if sub_parameters == parameters:
                break
            
            removing_files = []
            for sub_file in sub_files:
                if sub_file >= last_file:
                    removing_files.append(sub_file)
                else:
                    break
            
            if removing_files:
                sub_files = [file for file in sub_files if file not in removing_files]
                files_for_parameters[sub_parameters] = sub_files
                if not sub_files:
                    return odjects

    for parameters, files in files_for_parameters.items():
        current_object = _load_object_from_file_name(files[0])
        odjects[parameters.operation] = current_object
            
    return odjects
    
def get_last_temporary_file(parameters_temporary_file):
    file_names = get_temporary_files(parameters_temporary_file)
    if not file_names:
        return None
    
    file_names.sort()
    return file_names[-1]
    
def load_object_from_file(parameters_temporary_file):
    last_file = get_last_temporary_file(parameters_temporary_file)
    if last_file is None:
        return None
    
    return _load_object_from_file_name(last_file)
    
def _load_object_from_file_name(file_name): 
    dir_name = Settings.FOLDER_FOR_TEMPORARY_FILES
    full_path = dir_name + '/' + file_name
    with open(full_path, 'r') as f:
        return json.load(f)
       
def metaclass_resolver(*classes):
    metaclass = tuple(set(type(cls) for cls in classes))
    metaclass = metaclass[0] if len(metaclass)==1 \
                else type("_".join(mcls.__name__ for mcls in metaclass), metaclass, {}) 
    return metaclass("_".join(cls.__name__ for cls in classes), classes, {})

def str_to_int(number_str):
        number_int = 0
        for symbol in number_str:
            code_symbol = ord(symbol)
            if code_symbol<48 or code_symbol>57:
                break
            number_int = number_int*10 + (code_symbol - 48)
        return number_int
    
def _get_temporary_file_name(parameters_temporary_file, with_date_and_time):
    dir_name = Settings.FOLDER_FOR_TEMPORARY_FILES
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    parameters = list(parameters_temporary_file)
    if with_date_and_time:
        current_date_time = str(datetime.now())[:19]
        parameters.insert(0, current_date_time) 
       
    str_parameters  = map(str, parameters)
    file_name       = ', '.join(str_parameters)
    file_name      += '.json'
    return file_name 

def find_nearest_value_in_array(array, value):
    if type(array) not in  [list, tuple]:
        #single value, not collection of values
        return array
    
    if not array:
        #nothing to compare with
        return None
    
    if type(array[0]) not in [int, float]:   
        #types not compared for closeness to each other 
        if value in array:
            return value
        else:
            return None
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx] 

def concatenate_list_of_lists(list_of_lists):
    all_lists = []
    for curent_list in list_of_lists:
        all_lists.extend(curent_list)
    return all_lists  
