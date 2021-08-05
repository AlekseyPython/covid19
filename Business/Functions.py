from numpy import datetime64, timedelta64, unique
import Settings


def get_counted_unique_values_nd_array(np_data):
    unique_values = {}
    names = np_data.dtype.names
    for name in names:
        uniques, counts = unique(np_data[name], return_counts=True)
        for value, count in zip(uniques, counts):
            unique_values[value] = unique_values.get(value, 0) + count
    return unique_values

def get_counted_unique_values_dataframe(data):
    unique_values = {}
    for column in data.columns:
        unique_values_column = data[column].value_counts()
        for unique_value, quantity in unique_values_column.items():
            unique_values[unique_value] = unique_values.get(unique_value, 0) + quantity
                
    unique_values = {key:value for key,value in unique_values.items() if value>0}
    
    all_data = data.size * len(data.columns)
    all_values = sum(unique_values.values())
    if all_data > all_values:
        unique_values['NA'] = all_data - all_values
    
    return unique_values

def get_unique_values_dataframe(data):
    unique_values = ()
    for column in data.columns:
        unique_values += data[column].unique()
    return unique_values

def get_dict_values_in_percentage(dic):
    total_amount = 0
    for value in dic.values():
        total_amount += value
    
    new_dic = {}    
    for key, value in dic.items():
        new_dic[key] = round(100*value/total_amount, 1)
    return new_dic
        
def sort_dict_by_values(dic, forward_direction=True):
    if forward_direction:
        order = 1
    else:
        order = -1
    return {k: v for k, v in sorted(dic.items(), key=lambda item: order*item[1])}

def get_quantity_days_in_month(month): 
    return get_quantity_days_in_month.numbers[month]
get_quantity_days_in_month.numbers = dict(zip(range(1, 13), [30, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])) #culling 1 january

def _get_list_days_zodiacs():
    numbers = []
    numbers.append((datetime64('2020-01-20', 'D') - datetime64('2019-12-23', 'D'))/timedelta64(1, 'D'))#culling 1 january
    numbers.append(1 + (datetime64('2020-02-19', 'D') - datetime64('2020-01-21', 'D'))/timedelta64(1, 'D'))
    numbers.append(0.25 + (datetime64('2020-03-20', 'D') - datetime64('2020-02-20', 'D'))/timedelta64(1, 'D')) #leap year
    numbers.append(1 + (datetime64('2020-04-20', 'D') - datetime64('2020-03-21', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-05-21', 'D') - datetime64('2020-04-21', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-06-21', 'D') - datetime64('2020-05-22', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-07-22', 'D') - datetime64('2020-06-22', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-08-21', 'D') - datetime64('2020-07-23', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-09-23', 'D') - datetime64('2020-08-22', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-10-23', 'D') - datetime64('2020-09-24', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-11-22', 'D') - datetime64('2020-10-24', 'D'))/timedelta64(1, 'D'))
    numbers.append(1 + (datetime64('2020-12-22', 'D') - datetime64('2020-11-23', 'D'))/timedelta64(1, 'D'))
    return numbers
    
def get_quantity_days_in_zodiac(zodiac): 
        return get_quantity_days_in_zodiac.numbers[zodiac]
get_quantity_days_in_zodiac.numbers = dict(zip(Settings.VALUES_ZODIAC_SIGNS, _get_list_days_zodiacs()))   
            
