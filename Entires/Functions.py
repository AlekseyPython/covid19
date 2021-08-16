import sys
from .SimilarColumns import SimilarColumns

def get_variables_of_module(module, var_type, with_names=False, prefixes_exclude=[]):
    if with_names:
        variables = {}
    else:
        variables = []
        
    for var in module.__dict__.values():
        if type(var) != var_type:
            continue
        
        name = var.get_name()
        for prefix in prefixes_exclude:
            if name.startswith(prefix):
                break
        else:
            if with_names:
                variables[name] = var
            else:
                variables.append(var)
    return variables

def get_columns(converted=True, prefixes_exclude=[], with_names=False):
    if converted:
        module_name = 'Entires.ConvertedColumns'
    else:
        module_name = 'Entires.SourceColumns'
        
    module = sys.modules[module_name]
    var_type = module._Column
    return get_variables_of_module(module, var_type, with_names, prefixes_exclude)

def get_column_by_name(name, converted=True):
    columns = get_columns(converted=converted, with_names=True)
    return columns[name]

def get_columns_by_part_name(part_name, converted=True):
    all_columns = get_columns(converted=converted, with_names=True)
    return [name for name in all_columns.keys() if name.find(part_name) >= 0]
            
def get_similar_columns(converted=True):   
    if converted:
        module_name = 'Entires.ConvertedColumns'
    else:
        module_name = 'Entires.SourceColumns'
        
    module = sys.modules[module_name]
    var_type = SimilarColumns
    with_names=False
    prefixes_exclude=[]
    return get_variables_of_module(module, var_type, with_names, prefixes_exclude)    
    
