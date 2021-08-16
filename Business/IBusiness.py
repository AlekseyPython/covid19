import Settings


class IBusiness:
    def __init__(self):pass
    
    def create_moscow_regions_maps(self, signal_message):
        from Business.MoscowRegionsMapsTask import Task
        task = Task(signal_message)
        return self._perform_task(task)
        
    def convert_data_to_desired_types(self, signal_message):
        from Business.ConvertDataToDesiredTypesTask import Task
        task = Task(signal_message)
        return self._perform_task(task)
    
    def count_air_pollutions_for_moscow_citizens(self, signal_message, include_stations_near_the_roads):
        from Business.CountAirPollutionsForMoscowCitizensTask import Task
        task = Task(signal_message, include_stations_near_the_roads)
        return self._perform_task(task)
    
    def get_probability_according_to_parameter(self, signal_message, parameter, selections, fitting_curve):
        from Business.ProbabilityAccordingToParameterTask import Task
        task = Task(signal_message, parameter, selections, fitting_curve)
        return self._perform_task(task)
    
    def get_general_list_values(self, signal_message, parameter, source_of_data_is_converted):
        from Business.GeneralListValuesTask import Task
        task = Task(signal_message, parameter, source_of_data_is_converted)
        return self._perform_task(task)
    
    def tuning_prediction_model(self, signal_message, analysis_type , operation, severities_of_disease, print_only_final_results):
        from Business.TuningPredictionModelTask import Task
        task = Task(signal_message, analysis_type, operation, severities_of_disease, print_only_final_results)
        return self._perform_task(task)
    
    def get_probability_distribution(self, signal_message, parameter, selections):
        from Business.ProbabilityDistributionTask import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_time_dependence(self, signal_message, parameter, selections):
        from Business.TimeDependenceTask import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_spread_disease(self, signal_message, parameter, selections):
        from Business.SpreadOfDiseaseInMoscowTask import Task
        task = Task(signal_message, parameter, selections)
        return self._perform_task(task)
    
    def get_air_pollutions_for_moscow_citizens(self, signal_message, period_pollution, parameter_air_pollutin, selections):
        from Business.AirPollutionInMoscowTask import Task
        task = Task(signal_message, period_pollution, parameter_air_pollutin, selections)
        return self._perform_task(task)
    
    def _perform_task(self, task):
        if Settings.debuge_mode:
            task.run()
        else:
            task.start()
            task.wait()
        
        return task.result