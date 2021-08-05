import random
import math


class GridSearch:
    def __init__(self, counted_func, compare_metric, silently=True, quantity_continuously_bad_steps_before_break=2, quantity_all_bad_steps_before_break=5, 
                 quantity_shuffling_parameters=1, set_maximum_parameters=False):
        
        self.counted_func = counted_func
        self.compare_metric = compare_metric
        self.silently = silently
        self.quantity_continuously_bad_steps_before_break = quantity_continuously_bad_steps_before_break
        self.quantity_all_bad_steps_before_break = quantity_all_bad_steps_before_break
        self.quantity_shuffling_parameters = quantity_shuffling_parameters
        self.set_maximum_parameters = set_maximum_parameters
        
        #optimize counted func
        self.values_counted_func = {}
        
    def count(self, parameters):
        #first point
        if not self.silently:
            print('first point')
                
        best_parameters = self._get_values_parameters(parameters)
        best_result = self.counted_func(best_parameters)
        
        keys_shuffled_parameters = set()
        while len(keys_shuffled_parameters) < self.quantity_shuffling_parameters + 1: #1 iteration with source order without shuffling
            previous_result     = best_result
            previous_parameters = best_parameters.copy()
            first_iteration     = len(keys_shuffled_parameters)==0
        
            parameters, keys_shuffled_parameters = self._shuffle_parameters(parameters, keys_shuffled_parameters)
            if parameters is None:
                #the number of possible combinations of parameters has run out
                break
        
            best_parameters, best_result = self._optimize_all_parameters(parameters, best_parameters, best_result, first_iteration)
            if self.compare_metric(previous_result, best_result) == -1:
                raise RuntimeError('After shuffling of the parameters, a worse value of the resulting function was obtained!')
        
            if not self.silently:       
                print('previous_parameters=' + str(previous_parameters) + '\tprevious_result=' + str(previous_result))
                print('parameters=' + str(best_parameters) + '\tresult=' + str(best_result))
                
        return best_parameters, best_result
    
    def _shuffle_parameters(self, parameters, keys_shuffled_parameters):
        slave_parameters = self._get_slave_parameters(parameters)
        independent_keys = [key for key in parameters.keys() if key not in slave_parameters]
        
        quantity_combinations = math.factorial(len(independent_keys))
        while len(keys_shuffled_parameters) < quantity_combinations:
            if keys_shuffled_parameters:
                random.shuffle(independent_keys)
            else:
                #first iteration don't shuffle, because first order is important!
                pass
            
            independent_keys_tuple = tuple(independent_keys)
            if independent_keys_tuple not in keys_shuffled_parameters:
                keys_shuffled_parameters.add(independent_keys_tuple)
                break
        else:    
            return None, None
        
        independent_keys.extend(slave_parameters)
        shuffled_parameters = {key:parameters[key] for key in independent_keys}
        return shuffled_parameters, keys_shuffled_parameters
            
    def _get_slave_parameters(self, parameters):
        slave_parameters = []
        for characteristics in parameters.values():
            if 'slave_parameters' in characteristics:
                slave_parameters.extend(characteristics['slave_parameters'])
        return set(slave_parameters)
    
    def _optimize_all_parameters(self, parameters, best_parameters, best_result, first_iteration):
        slave_parameters = self._get_slave_parameters(parameters)
        for parameter, characteristics in parameters.items():
            if parameter in slave_parameters:
                continue
            
            previous_result = best_result
            if 'slave_parameters' not in characteristics:
                best_parameters, best_result = self._optimize_parameter(best_parameters, best_result, parameter, characteristics, first_iteration)
            
            else:
                points = self._get_points(characteristics)
                slave_parameters = characteristics['slave_parameters']
                for point, slave_parameter in zip(points, slave_parameters):
                    master_parameter            = {parameter: point}
                    slave_characteristics       = parameters[slave_parameter]
                    best_parameters, best_result= self._optimize_parameter(best_parameters, best_result, slave_parameter, slave_characteristics, first_iteration, master_parameter)
            
            if self.compare_metric(previous_result, best_result) == -1:
                raise RuntimeError('After optimization of the parameter:' + str(parameter) + ' a worse value of the resulting function was obtained!')
                
        return best_parameters, best_result
    
    def _count_func(self, master_parameter, parameter, point, current_parameters): 
        if master_parameter is None:
            common_parameter = str(parameter)
        else:
            common_parameter = str(master_parameter) + '/' + str(parameter)
            
        if common_parameter not in self.values_counted_func:
            self.values_counted_func.clear()
            self.values_counted_func[common_parameter] = {}
            
        if point in self.values_counted_func[common_parameter]:
            return self.values_counted_func[common_parameter][point]
        
        current_parameters[parameter] = point
        current_result = self.counted_func(current_parameters)
        
        self.values_counted_func[common_parameter][point] = current_result
        return current_result
                
    def _optimize_parameter(self, best_parameters, best_result, parameter, characteristics, first_iteration, master_parameter=None):
        points = self._get_points(characteristics)
        start_point = best_parameters[parameter]
        
        if type(start_point) == str:
            this_is_quality_feature = True
        else:
            this_is_quality_feature = False
        
        #create auxiliary variable used only for calculation 
        current_parameters = best_parameters.copy()
        if master_parameter is not None:
            current_parameters.update(master_parameter)
        
        point_for_parameter = None
        result_for_parameter = None
        previous_result = None
        set_maximum_parameter = characteristics.get('set_maximum_parameter', self.set_maximum_parameters)
        
        quantity_continuously_bad_steps = 0
        for point in points:
            if not first_iteration:
                #search near previous optimum point
                index_start = points.index(start_point)
                current_index = points.index(point)
                if abs(index_start-current_index) > self.quantity_continuously_bad_steps_before_break:
                    continue
                
            if not self.silently:
                if master_parameter is None:
                    print(str(parameter) + ' = ' + str(point))
                    
                else:
                    key_master = list(master_parameter.keys())[0]
                    value_master = master_parameter[key_master]
                    print(str(key_master) + '/' + str(parameter) + ' = ' + str(value_master) + ' / ' + str(point))
                
            if point == start_point:
                if master_parameter is None or master_parameter.items() <= best_parameters.items():
                    current_result = best_result
                else:
                    current_result = self._count_func(master_parameter, parameter, point, current_parameters)
                    
            else:
                current_result = self._count_func(master_parameter, parameter, point, current_parameters)
                
            if result_for_parameter is None:
                quantity_all_bad_steps = 0
                result_for_parameter = current_result
                point_for_parameter = point
                
            elif self.compare_metric(result_for_parameter, current_result) == 1:
                quantity_all_bad_steps = 0
                result_for_parameter = current_result
                point_for_parameter = point
            
            elif self.compare_metric(result_for_parameter, current_result) == 0:
                if not set_maximum_parameter:
                    quantity_all_bad_steps += 0.5
                
                if point > point_for_parameter and set_maximum_parameter:
                    point_for_parameter = point
                        
            else:
                quantity_all_bad_steps += 1
            
            if this_is_quality_feature:
                #count all values, ignoring bad values 
                continue
                    
            if quantity_all_bad_steps >= self.quantity_all_bad_steps_before_break:
                break
            
            if previous_result is not None:
                if self.compare_metric(previous_result, current_result) == 1:    
                    quantity_continuously_bad_steps = 0
                    
                elif self.compare_metric(previous_result, current_result) == 0:
                    if not set_maximum_parameter:
                        quantity_continuously_bad_steps += 0.5
                        
                else:
                    quantity_continuously_bad_steps += 1
                    if quantity_continuously_bad_steps >= self.quantity_continuously_bad_steps_before_break:
                        break
                
            previous_result = current_result
        
        if self.compare_metric(best_result, result_for_parameter) == 1:
            best_result = result_for_parameter
            best_parameters[parameter] = point_for_parameter
            if master_parameter is not None:
                best_parameters.update(master_parameter)
                
        elif self.compare_metric(best_result, result_for_parameter) == 0 and set_maximum_parameter and point_for_parameter > best_parameters[parameter]:
            best_parameters[parameter] = point_for_parameter
            if master_parameter is not None:
                best_parameters.update(master_parameter)
            
        return best_parameters, best_result
        
    @staticmethod    
    def _get_values_parameters(parameters):
        current_parameters = {}
        for parameter, characteristics in parameters.items():
            if 'start' in characteristics:
                current_parameters[parameter] = characteristics['start']
                
            elif 'values' in characteristics:
                current_parameters[parameter] = characteristics['values'][0]
                
            else:
                current_parameters[parameter] = characteristics['min_value']
        return current_parameters
        
    @staticmethod
    def _get_points(characteristics):
        if 'values' in characteristics:
            return characteristics['values']
        
        else:
            if 'slave_parameter' in characteristics:
                raise RuntimeError('Slave- parameters are implemented only for parameters specified by list of values!')
                
        values = []
        value = characteristics['min_value']
        while value <= characteristics['max_value']:
            values.append(value)
            if 'step' in characteristics:
                value = value + characteristics['step']
                
            elif 'multiplier' in characteristics:
                value = value * characteristics['multiplier']
        return values