class IBusiness:
    def __init__(self):pass
    
    def create_moscow_regions_maps(self, signal_message):
        from .MoscowRegionsMaps import Task
        task = Task(signal_message)
        return self._perform_task(task)
        
    def convert_data_to_desired_types(self, signal_message):
        from .ConvertDataToDesiredTypes import Task
        task = Task(signal_message)
        return self._perform_task(task)
    
    def count_air_pollutions_for_moscow_citizens(self, signal_message, include_stations_near_the_roads):
        from .CountAirPollutionsForMoscowCitizens import Task
        task = Task(signal_message, include_stations_near_the_roads)
        return self._perform_task(task)
    
    def get_probability_according_to_parameter(self, signal_message, parameter, selections, fitting_curve):
        from .ProbabilityAccordingToParameter import Task
        task = Task(signal_message, parameter, selections, fitting_curve)
        return self._perform_task(task)
    
    def get_general_list_values(self, signal_message, parameter, source_of_data_is_converted):
        from .GeneralListValues import Task
        task = Task(signal_message, parameter, source_of_data_is_converted)
        return self._perform_task(task)
    
    def tuning_prediction_model(self, signal_message, analysis_type , operation, model_type, severities_of_disease, print_only_final_results):
        from .TuningPredictionModel import Task
        task = Task(signal_message, analysis_type, operation, model_type, severities_of_disease, print_only_final_results)
        return self._perform_task(task)
    
    def get_probability_distribution(self, signal_message, parameter, selections):
        from .ProbabilityDistribution import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_time_dependence(self, signal_message, parameter, selections):
        from .TimeDependence import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_spread_disease(self, signal_message, parameter, selections):
        from .SpreadOfDiseaseInMoscow import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_air_pollutions_for_moscow_citizens(self, signal_message, period_pollution, parameter_air_pollutin, selections):
        from .AirPollutionInMoscow import Task
        task = Task(signal_message, period_pollution, parameter_air_pollutin, selections)
        return self._perform_task(task)
    
    def _perform_task(self, task):
        # if Settings.debuge_mode:
        task.run()
        # else:
        #     task.start()
        #     task.wait()
        
        return task.result