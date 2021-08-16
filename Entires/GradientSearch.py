from sklearn.utils import shuffle


class GradientSearch:
    def __init__(self, counted_func, compare_metric, random_state, silently=True, shuffle=True, quantity_all_bad_steps_before_break=2,  
                 quantity_continuously_steps_on_plateau=3, set_maximum_parameters=False):
        
        self.counted_func = counted_func
        self.compare_metric = compare_metric
        self.random_state = random_state
        self.silently = silently
        self.shuffle = shuffle
        self.quantity_all_bad_steps_before_break = quantity_all_bad_steps_before_break
        self.quantity_continuously_steps_on_plateau = quantity_continuously_steps_on_plateau
        self.set_maximum_parameters = set_maximum_parameters
        
        #optimize counted func
        self.values_counted_func = {} 
        
    def count(self, parameters):
        parameters = self._shuffle_parameters(parameters)
        
        #first point
        best_parameters = self._get_values_parameters(parameters)
        if not self.silently:
            print('first point: ' + str(best_parameters))
            
        best_result  = self.counted_func(best_parameters)
        
        start_result     = best_result
        start_parameters = best_parameters.copy()
        
        slave_parameters = self._get_slave_parameters(parameters)
        for parameter, characteristics in parameters.items():
            if parameter in slave_parameters:
                continue
        
            previous_result = best_result
            if 'slave_parameters' not in characteristics:
                best_parameters, best_result = self._count_parameter(best_parameters, best_result, parameter, characteristics)
        
            else:
                for direction in [-1, 1]:
                    if 'values' not in characteristics:
                        raise RuntimeError('Slave- parameters are implemented only for parameters specified by list of values!')
        
                    master_values = characteristics['values']
                    slave_parameters = characteristics['slave_parameters']
                    points = self._get_points(characteristics, direction, multiplier_first_step=1)
                    for point in points:
                        index_point = master_values.index(point)
                        slave_parameter = slave_parameters[index_point]
                        master_parameter = {parameter: point}
                        slave_characteristics = parameters[slave_parameter]
                        best_parameters, best_result = self._count_parameter(best_parameters, best_result, slave_parameter, slave_characteristics, master_parameter)
        
            if self.compare_metric(previous_result, best_result) < 0:
                raise RuntimeError('After optimization of the parameter:' + str(parameter) + ', a worse value of the resulting function was obtained!')
        
        if not self.silently:       
            print('start parameters=' + str(start_parameters) + '\start result=' + str(start_result))
            print('current parameters=' + str(best_parameters) + '\tcurrent result=' + str(best_result))
        
        increment_metric = self.compare_metric(start_result, best_result)    
        return best_parameters, best_result, increment_metric
    
    def _get_slave_parameters(self, parameters):
        slave_parameters = []
        for characteristics in parameters.values():
            if 'slave_parameters' in characteristics:
                slave_parameters.extend(characteristics['slave_parameters'])
        return set(slave_parameters)
    
    def _shuffle_parameters(self, parameters):
        if not self.shuffle:
            return parameters
        
        l = list(parameters.items())
        l = shuffle(l, random_state=self.random_state)
        self.random_state += 1
        parameters = dict(l)
        return parameters
    
    @staticmethod    
    def _get_values_parameters(parameters):
        values_parameters = {}
        for parameter, characteristics in parameters.items():
            if 'value' in characteristics:
                values_parameters[parameter] = characteristics['value']
            else:
                values_parameters[parameter] = characteristics['start']
        return values_parameters
    
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
        
    def _count_parameter(self, best_parameters, best_result, parameter, characteristics, master_parameter=None):
        #create auxiliary variable used only for calculation 
        current_parameters = best_parameters.copy()
        if master_parameter is not None:
            current_parameters.update(master_parameter)
            
        start_result = best_result
        start_point = best_parameters[parameter]
            
        if type(start_point) == str:
            this_is_quality_feature = True
            if 'values' not in characteristics:
                raise RuntimeError('Feature string values can only be set using field "values"!')
            
            index_start_point     = characteristics['values'].index(start_point)
            last_index            = len(characteristics['values']) - 1
            max_lenght_first_step = max(index_start_point, last_index - index_start_point)
        else:
            this_is_quality_feature = False
            max_lenght_first_step = self.quantity_all_bad_steps_before_break
            
        set_maximum_parameter = characteristics.get('set_maximum_parameter', self.set_maximum_parameters)
        for multiplier_first_step in range(1, max_lenght_first_step + 1):
            if best_parameters[parameter] != start_point:
                break
            
            quantity_steps_on_plateau = 0 
            for direction in [-1, 1]:
                point_for_direction = None
                result_for_direction = None
                
                if quantity_steps_on_plateau >= self.quantity_continuously_steps_on_plateau and not this_is_quality_feature:
                    break
                            
                for point in self._get_points(characteristics, direction, multiplier_first_step):
                    if not self.silently:
                        if master_parameter is None:
                            print(str(parameter) + ' = ' + str(point))
                            
                        else:
                            key_master = list(master_parameter.keys())[0]
                            value_master = master_parameter[key_master]
                            print(str(key_master) + '/' + str(parameter) + ' = ' + str(value_master) + ' / ' + str(point))
                            
                    if point == start_point:
                        if master_parameter is None or master_parameter.items() <= current_parameters.items():
                            current_result = start_result
                        else:
                            current_result = self._count_func(master_parameter, parameter, point, current_parameters)
                            
                    else:
                        current_result = self._count_func(master_parameter, parameter, point, current_parameters)
                        
                    if result_for_direction is None:
                        point_for_direction = point
                        result_for_direction = current_result
                        
                    elif self.compare_metric(result_for_direction, current_result) > 0:
                        quantity_steps_on_plateau = 0
                        result_for_direction = current_result
                        point_for_direction = point
                        
                    elif self.compare_metric(result_for_direction, current_result) == 0:
                        if point > point_for_direction and set_maximum_parameter:
                            point_for_direction = point
                    
                    elif this_is_quality_feature:
                        #count all values, ignoring bad values 
                        continue
                        
                    elif self.compare_metric(result_for_direction, current_result) == 0:    
                        if (set_maximum_parameter and direction==-1) or (not set_maximum_parameter and direction==1):
                            quantity_steps_on_plateau += 1
                            if quantity_steps_on_plateau >= self.quantity_continuously_steps_on_plateau:
                                break
                        
                    else:
                        break
                        
                if self.compare_metric(best_result, result_for_direction) > 0:
                    best_result = result_for_direction
                    best_parameters[parameter] = point_for_direction
                    if master_parameter is not None:
                        best_parameters.update(master_parameter)
                        
                elif self.compare_metric(best_result, result_for_direction) == 0 and set_maximum_parameter and point_for_direction > best_parameters[parameter]:
                    best_parameters[parameter] = point_for_direction
                    if master_parameter is not None:
                        best_parameters.update(master_parameter)
                        
        return best_parameters, best_result
                        
    def _get_points(self, characteristics, direction, multiplier_first_step):
        was_first_step = False
        if 'values' in characteristics:
            values = characteristics['values']
            index = values.index(characteristics['start'])
            
            while index>=0 and index<len(values):
                value = values[index]
                yield value
                if was_first_step:
                    muliplier = 1
                else:
                    muliplier = multiplier_first_step
                    was_first_step = True
                    
                index += muliplier * direction
                
        else:
            if 'slave_parameter' in characteristics:
                raise RuntimeError('Slave- parameters are implemented only for parameters specified by list of values!')
            
            value = characteristics['start']
            while value>=characteristics['min_value'] and value<=characteristics['max_value']:
                yield value
                
                if was_first_step:
                    muliplier = 1
                else:
                    muliplier = multiplier_first_step
                    was_first_step = True
                    
                if 'step' in characteristics:
                    value = value + direction * muliplier * characteristics['step']
                    
                elif 'multiplier' in characteristics:
                    value = value * pow(characteristics['multiplier'], direction*muliplier)
    