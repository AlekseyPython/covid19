import time
import Functions

def performance_meter(func):
    def performance_meter_dev(*args, **kwargs) -> float:
        initial_time = time.time()
        return_value = func(*args, **kwargs)
        elapsed_time = round(time.time() - initial_time, 2)
        print('Time elapsed for ' + func.__name__ + ' = ' + str(elapsed_time))
        return return_value
            
    return performance_meter_dev


def save_and_load(func):
    def save_and_load_dev(*args, **kwargs):
        file_name = func.__name__
        if args:
            if str.find(str(args[0]), 'object'):
                file_name += str(args[1:])
            else:
                file_name += str(args)
                
        if kwargs:
            file_name += str(dict(sorted(kwargs.items())))
                
        return_value = Functions.load_object_from_file(file_name)
        if return_value is None:
            return_value = func(*args, **kwargs)
            
        Functions.save_object_to_file(return_value, file_name)    
        return return_value
            
    return save_and_load_dev

