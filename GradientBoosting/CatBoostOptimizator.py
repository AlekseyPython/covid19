import functools
import statistics
import numpy as np
from collections import Counter
from Business.AMachineLearning import AOptimizatorGradientBoosting
from .Optimizator import Optimizator
from .LearningData import Features, PreparedData, LearningData
from .CatBoostModel import Pools, CatBoostModel
from Entires import Enums
from Entires.GridSearch import GridSearch
from Entires.GradientSearch import GradientSearch
import Settings, Functions


cat_boost = Enums.TypesOfGradientBoostingModel.CatBoost.value

class CatBoostOptimizator(AOptimizatorGradientBoosting, Optimizator):
    def __init__(self, analysis_type, source_data, converted_data, air_pollution, calculation_expediency_threshold, limit_unresulting_calculations):
        AOptimizatorGradientBoosting.__init__(self)
        Optimizator.__init__(self)
        
        self.analysis_type = analysis_type
        self.calculation_expediency_threshold = calculation_expediency_threshold
        self.limit_unresulting_calculations = limit_unresulting_calculations
        
        np.random.seed(42)
        
        #this coefficient shows, that the error to classify a weaker course of the disease as a stronger one
        #is more important for us, than the reverse
        self.multiplier_of_zero_class   = 2
        
        self.common_penalty             = pow(10, -8)
        self.default_train_size         = 0.8
        self.default_learning_rate      = 0.05
        
        #tolerance coefficient overfitting, when the result is improved. an increase in this coefficient leads to an increase in randomness,
        #and the parameters of model may be chosen incorrectly
        self.ratio_of_importance_optimization_to_overfitting = 1
        
        if Settings.debuge_mode:
            self.quantity_iterations_for_stable_parameters                  = 1
            self.quantity_iterations_for_stable_eliminated_features         = 1
            self.quantity_iterations_for_stable_common_penalty              = 1
            self.quantity_iterations_for_stable_penalty                     = 1
            self.quantity_iterations_for_stable_penalties_coefficient       = 1
            self.quantity_iterations_for_stable_train_test_split            = 1
            self.quantity_iterations_for_resulting_count                    = 4
            self.quantity_iterations_for_stable_learning_rate               = 1
            
        else:
            self.quantity_iterations_for_stable_parameters                  = 25 
            self.quantity_iterations_for_stable_eliminated_features         = 10
            self.quantity_iterations_for_stable_common_penalty              = 25
            self.quantity_iterations_for_stable_penalty                     = 5
            self.quantity_iterations_for_stable_penalties_coefficient       = 25
            self.quantity_iterations_for_stable_train_test_split            = 25
            self.quantity_iterations_for_resulting_count                    = 35
            self.quantity_iterations_for_stable_learning_rate               = 25
        
        self.increasing_percent_iterations_for_gradient_mode = 30
        self.random_numbers_for_resulting_count = self._genearte_randoms(self.quantity_iterations_for_resulting_count)
        
        self.features = Features(analysis_type, air_pollution)
        self.prepared_data = PreparedData(analysis_type, self.features, source_data, converted_data, air_pollution)
    
    def regularized_metric(self, train, test, metric='Logloss'):
        #regularize by the overfitting
        if metric ==  'Logloss':
            return test + self.ratio_of_importance_optimization_to_overfitting * (test-train)
        else:
            return test - self.ratio_of_importance_optimization_to_overfitting * (train -test)
            
    @staticmethod
    def _compare_metric(first, second):
        #use logloss (cross entropy), for which the less the better
        return round(first-second, 3)
        
    def _create_learning_data(self, severity_of_disease, random_state=42, train_size=None, eliminated_features=[]):
        if train_size is None:
            train_size = self.default_train_size
            
        return LearningData(self.prepared_data, self.features, severity_of_disease, random_state, train_size, eliminated_features)
    
    def _create_model(self, current_features, pools, random_state, params_model={}, penalties={}, common_penalty=None, fit_model=True):
        if common_penalty is None:
            common_penalty = self.common_penalty
        
        #this parameter is set by the customer, therefore it does not vary and always has the same value
        params_model.update({'class_weights': [self.multiplier_of_zero_class, 1]})  
          
        model = CatBoostModel(current_features, pools, random_state, params_model, penalties, common_penalty)
        if fit_model: model.fit()
        return model
    
    def _genearte_randoms(self, quantity):
        if Settings.debuge_mode:
            np.random.seed(42)
            
        return np.random.randint(low=0, high=10000, size=quantity)
        
    def _get_saved_parameters(self, severity_of_disease):
        all_params = {}
        operations = Enums.OperationsOfMachineLearning
        
        all_params['parameters_avoidance_overfitting'] = self.get_saved_result(operations.AvoidanceOverfitting.value, cat_boost, severity_of_disease)
        if all_params['parameters_avoidance_overfitting'] is None:
            all_params['parameters_avoidance_overfitting'] = {}
        
        all_params['eliminated_features'] = self.get_saved_result(operations.SearchEliminatedFeatures.value, cat_boost, severity_of_disease)
        if all_params['eliminated_features'] is None:
            all_params['eliminated_features'] = {}
            
        all_params['penalties'] = self.get_saved_result(operations.SearchPenalties.value, cat_boost, severity_of_disease)
        if all_params['penalties'] is None:
            all_params['penalties'] = {}
         
        all_params['hyperparameters'] = self.get_saved_result(operations.SearchHyperparameters.value, cat_boost, severity_of_disease)
        if all_params['hyperparameters'] is None:
            all_params['hyperparameters'] = {}
        
        all_params['train_size'] = self.get_saved_result(operations.OptimizeTrainTestSplit.value, cat_boost, severity_of_disease)
        if all_params['train_size'] is None:
            all_params['train_size'] = {}
            
        return all_params
    
    def features_profiling(self, severity_of_disease, silently=True):
        learning_data = self._create_learning_data(severity_of_disease)
        return learning_data.profile_report(silently)
    
    def _get_increasing_quantity_for_gradient_mode(self, quantity, exist_previous_iteration):
        if exist_previous_iteration:
            quantity *= (1 + self.increasing_percent_iterations_for_gradient_mode/100)
            quantity = round(quantity)
        return quantity
    
    def _get_number_iteration(self, operation, severity_of_disease): 
        parameters_temporary_file   = Enums.ParametersTemporaryFiles(self.analysis_type, operation, cat_boost, severity_of_disease)
        return Functions.count_quantity_files(parameters_temporary_file)
                   
    def jump_operation(self, operation, severity_of_disease):
        #the procedure is needed for repeatability of the result after partial spreading
        number_iteration = self._get_number_iteration(operation, severity_of_disease)
        
        operations = Enums.OperationsOfMachineLearning
        if operation == operations.AvoidanceOverfitting.value:
            initial_quantities    = [self.quantity_iterations_for_stable_parameters]
            
        elif operation == operations.SearchEliminatedFeatures.value:
            initial_quantities    = [self.quantity_iterations_for_stable_eliminated_features]
            
        elif operation == operations.SearchPenalties.value:
            initial_quantities    = [self.quantity_iterations_for_stable_penalty, self.quantity_iterations_for_stable_penalties_coefficient]
            if number_iteration == 1:
                initial_quantities.append(self.quantity_iterations_for_stable_common_penalty)
                
        elif operation == operations.SearchHyperparameters.value:
            initial_quantities    = [self.quantity_iterations_for_stable_parameters]
            
        elif operation == operations.OptimizeTrainTestSplit.value:
            initial_quantities    = [self.quantity_iterations_for_stable_train_test_split]
            
        else:
            raise RuntimeError("This function isn't intended for the passed operation type!")
        
        was_previous_iteration = (number_iteration >= 2)
        for current_quantity in initial_quantities:
            current_quantity = self._get_increasing_quantity_for_gradient_mode(current_quantity, was_previous_iteration)    
            self._genearte_randoms(current_quantity)    
    
    def get_previous_result_for_passing_calculation(self, operation, exist_parameters):
        significant_result = False
        metrics            = {'logloss': None}
        parameters         = exist_parameters | metrics | {'quantity_unresulting_calculations': self.limit_unresulting_calculations}
        result_operation   = Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, significant_result)
        return result_operation
    
    def clear_parameters_from_metrics_and_quantity_unresulting_calculations(self, loaded_params): 
        for name_parameters in loaded_params.keys():
            loaded_params[name_parameters].pop('logloss', None)
            loaded_params[name_parameters].pop('quantity_unresulting_calculations', None)
        return loaded_params
               
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.AvoidanceOverfitting.value, cat_boost)  
    def avoidance_overfitting(self, severity_of_disease, silently=True):
        operation = Enums.OperationsOfMachineLearning.AvoidanceOverfitting.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params    = self._get_saved_parameters(severity_of_disease)
        exist_parameters = loaded_params.pop('parameters_avoidance_overfitting')
        loaded_params    = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
        
        exist_parameters.pop('logloss', None)
        quantity_unresulting_calculations = exist_parameters.pop('quantity_unresulting_calculations', 0)
        if quantity_unresulting_calculations >= self.limit_unresulting_calculations:
            return self.get_previous_result_for_passing_calculation(operation, exist_parameters) 
        
        quantity_iterations = self.quantity_iterations_for_stable_parameters
        quantity_iterations = self._get_increasing_quantity_for_gradient_mode(quantity_iterations, exist_parameters)
        
        name_variabling_params = 'parameters_avoidance_overfitting'
        random_numbers         = self._genearte_randoms(quantity_iterations)
        counted_func           = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, name_variabling_params)
        
        parameters = {}
        quantity_shuffling_parameters = 1
        if exist_parameters:
            parameters['early_stopping_rounds'] = {'min_value':3,       'max_value':30, 'step':3}
            parameters['l2_leaf_reg']           = {'min_value':0,       'max_value':10, 'step':1, 'set_maximum_parameter':True}
            parameters['depth']                 = {'min_value':1,       'max_value':10, 'step':1}
            parameters['boosting_type']         = {'values':['Ordered', 'Plain']}
        
            for parameter, value in exist_parameters.items():
                parameters[parameter]['start'] = value
            
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state = quantity_shuffling_parameters + number_iteration - 1
            searcher = GradientSearch(counted_func, self._compare_metric, random_state, silently)
    
        else:
            parameters['early_stopping_rounds'] = {'min_value':5,       'max_value':20, 'step':5,   'start':5}
            parameters['l2_leaf_reg']           = {'min_value':2,       'max_value':8,  'step':2,   'start':8, 'set_maximum_parameter':True}
            parameters['depth']                 = {'min_value':3,       'max_value':8,  'step':2,   'start':5}
            parameters['boosting_type']         = {'values':['Ordered', 'Plain'], 'start':'Ordered'}
            
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters)
        
        result = searcher.count(parameters)
        
        increment_metric = result[2]
        if increment_metric < self.calculation_expediency_threshold:
            quantity_unresulting_calculations += 1
        else:
            quantity_unresulting_calculations = 0
        
        need_to_continue_optimization = (quantity_unresulting_calculations < self.limit_unresulting_calculations)
            
        metrics         = {'logloss': result[1]}
        parameters      = result[0] | metrics | {'quantity_unresulting_calculations': quantity_unresulting_calculations}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, need_to_continue_optimization)
        return result_operation 
        
    def _count_model_for_point(self, severity_of_disease, random_numbers, silently, loaded_params, metric_func, name_variabling_params, variabling_params):
        #don't use the optimal saved value during model tuning, in order to reduce processor time
        learning_rate = {'learning_rate': self.default_learning_rate}
        
        #saved in files varaibles
        common_penalty          = None
        separated_penalties     = None
        penalties_coefficient   = None
        train_size              = None
        
        if name_variabling_params == 'common_penalty':
            common_penalty = variabling_params['common_penalty']
        
        elif name_variabling_params == 'separated_penalties':
            separated_penalties = variabling_params
            
        elif name_variabling_params == 'penalties_coefficient':
            penalties_coefficient = variabling_params
            
        elif name_variabling_params == 'learning_rate':
            learning_rate = variabling_params
        
        elif name_variabling_params == 'train_size':
            train_size = variabling_params['train_size']

        else:   
            loaded_params[name_variabling_params] = variabling_params
        
        #restore saved in files variables
        penalties = loaded_params['penalties']
        if common_penalty is None:
            if penalties.get('common_penalty', False):
                common_penalty = penalties['common_penalty']
            else:
                common_penalty = self.common_penalty
                
        if separated_penalties is None:
            if penalties.get('separated_penalties', False):
                separated_penalties = penalties['separated_penalties']
            else:
                separated_penalties = {}
                
        if penalties_coefficient is None:
            if penalties.get('penalties_coefficient', False):
                penalties_coefficient = penalties['penalties_coefficient']
            else:
                penalties_coefficient = 1
            penalties_coefficient = {'penalties_coefficient': penalties_coefficient}
        
        if train_size is None:
            if loaded_params.get('train_size', False):
                train_size = loaded_params['train_size']['train_size']
            else:
                train_size = self.default_train_size
        
        if loaded_params.get('eliminated_features', False):
            eliminated_features = loaded_params['eliminated_features']['eliminated_features']
        else:
            eliminated_features = []
        
        #function 'get_feature_importance' work bad, when there's a penalty
        if name_variabling_params == 'quantity_unimportance_features':
            common_penalty          = 0
            separated_penalties     = {}
            penalties_coefficient   = {'penalties_coefficient': 1}
                  
        metrics = [] 
        all_unimportance_features = set()
        for random_number in random_numbers:
            learning_data = self._create_learning_data(severity_of_disease, random_number, train_size, eliminated_features)
            current_features = learning_data.get_current_features()
            pools = Pools(current_features, learning_data)
            
            all_params = loaded_params['parameters_avoidance_overfitting'] | loaded_params['hyperparameters'] | learning_rate | penalties_coefficient
            all_params = self._remove_conflicting_hyperparameters(all_params)
            
            model = self._create_model(current_features, pools, random_number, all_params, separated_penalties, common_penalty)
            
            best_score  = model.get_best_score()
            train_metric= round(100*best_score['learn']['Logloss'], 3)
            test_metric = round(100*best_score['validation']['Logloss'], 3)
            
            common_metric = metric_func(train_metric, test_metric)
            metrics.append(common_metric)
            
            if name_variabling_params == 'quantity_unimportance_features':
                importances = model.get_feature_importance()
                current_unimportance_features = importances[importances['Importances'] == 0]
                all_unimportance_features.update(current_unimportance_features['Feature Id'])
                
        metric = self.mean_metric(metrics)
        if not silently:
            if name_variabling_params == 'separated_penalties':
                print('unregularizated logloss=' + str(metric))
            else:
                print('regularizated logloss=' + str(metric))
            
        if name_variabling_params == 'quantity_unimportance_features':
            quantity_unimportance_features = len(all_unimportance_features)    
            return quantity_unimportance_features 
            
        return metric
    
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.SearchEliminatedFeatures.value, cat_boost)                      
    def search_eliminated_features(self, severity_of_disease, silently=True):
        def get_eliminated_features_parameters():
            silently                       = True
            name_variabling_params         = 'quantity_unimportance_features'
            quantity_unimportance_features = self._count_model_for_point(severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, name_variabling_params, variabling_params={})
            quantity_all_features          = len(self.features.get_all_features())
            
            parameters = {}
            parameters['quantity_unimportance_features'] = {'min_value':quantity_unimportance_features, 'max_value':quantity_all_features, 'step':1}
            return parameters
        
        operation = Enums.OperationsOfMachineLearning.SearchEliminatedFeatures.value    
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        #we use few random points for the calculation, because it is a very long procedure 
        #therefore, for the stability of the solution, we use the metric without regularization
        loaded_params             = self._get_saved_parameters(severity_of_disease)
        exist_eliminated_features = loaded_params.pop('eliminated_features') 
        loaded_params             = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
        
        exist_eliminated_features.pop('logloss', None)
        quantity_unresulting_calculations = exist_eliminated_features.pop('quantity_unresulting_calculations', 0)
        if quantity_unresulting_calculations >= self.limit_unresulting_calculations:
            return self.get_previous_result_for_passing_calculation(operation, exist_eliminated_features) 
        
        quantity_iterations = self.quantity_iterations_for_stable_eliminated_features
        quantity_iterations = self._get_increasing_quantity_for_gradient_mode(quantity_iterations, exist_eliminated_features)
        
        return_metrics      = True
        random_numbers      = self._genearte_randoms(quantity_iterations)
        counted_func = functools.partial(self._search_eliminated_features_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, return_metrics)
        
        quantity_shuffling_parameters = 0
        parameters = get_eliminated_features_parameters()
        if exist_eliminated_features:
            start = len(exist_eliminated_features['eliminated_features'])
            if start > parameters['quantity_unimportance_features']['max_value']:
                start = parameters['quantity_unimportance_features']['max_value']
                
            #it's better to change the minimum point, because this creates the possibility of incremental improvements 
            #If we change the starting point, then we will always get great randomness
            #the minimum point will be shifted by 2 relative to the start, so that both directions can be varied
            current_min_value = parameters['quantity_unimportance_features']['min_value']
            if start - current_min_value < 1:
                parameters['quantity_unimportance_features']['min_value'] =  max(0, start - 1)
                
            parameters['quantity_unimportance_features']['start'] = start
            
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state = quantity_shuffling_parameters + number_iteration - 1
            searcher = GradientSearch(counted_func, self._compare_metric, random_state, silently, set_maximum_parameters=True)
        else:
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters, 
                                  quantity_continuously_bad_steps_before_break=3, set_maximum_parameters=True)
        
        result = searcher.count(parameters)
        
        #get names eliminated features instead quantity
        return_metrics            = False
        num_features_to_eliminate = result[0]
        eliminated_features_names = self._search_eliminated_features_for_point(severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, return_metrics, num_features_to_eliminate)
        
        increment_metric = result[2]
        if increment_metric < self.calculation_expediency_threshold:
            quantity_unresulting_calculations += 1
        else:
            quantity_unresulting_calculations = 0
        
        need_to_continue_optimization = (quantity_unresulting_calculations < self.limit_unresulting_calculations)
        
        metrics         = {'logloss': result[1]}
        parameters      = {'eliminated_features': eliminated_features_names} | metrics | {'quantity_unresulting_calculations': quantity_unresulting_calculations}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, need_to_continue_optimization)
        
        return result_operation
    
    def _search_eliminated_features_for_point(self, severity_of_disease, random_numbers, silently, loaded_params, metric_func, return_metrics, num_features_to_eliminate):
        #select_features doesn't work with penalty! Error CatBoost! 
        learning_rate = {'learning_rate': self.default_learning_rate}
        if loaded_params['train_size']:
            train_size = loaded_params['train_size']['train_size']
        else:
            train_size = self.default_train_size
        
        metrics = []
        eliminated_features_names = []
        for random_number in random_numbers:
            learning_data   = self._create_learning_data(severity_of_disease, random_number, train_size=train_size)
            current_features= learning_data.get_current_features()
            pools           = Pools(current_features, learning_data)
            
            #function 'select_features' doesn't work with penalty! Error CatBoost! 
            all_params      = loaded_params['parameters_avoidance_overfitting'] | loaded_params['hyperparameters'] | learning_rate
            all_params      = self._remove_conflicting_hyperparameters(all_params)
            model           = self._create_model(current_features, pools, random_number, all_params, common_penalty=0, fit_model=False)
            result_selection= model.select_features(num_features_to_eliminate['quantity_unimportance_features'])
            
            if return_metrics:
                #after feature selection metrics don't count!
                train_metric, test_metric = model.eval_metrics(metrics=['Logloss'])
                train_metric= round(100 * train_metric['Logloss'][-1], 3)
                test_metric = round(100 * test_metric['Logloss'][-1], 3)
                
                common_metric = metric_func(train_metric, test_metric)
                metrics.append(common_metric)
                
            else:
                eliminated_features_names.extend(result_selection['eliminated_features_names'])
            
        if return_metrics:
            metric = self.mean_metric(metrics)
            if not silently:
                print('regularizated logloss=' + str(metric))
            return metric
        
        else:
            counter_features = Counter(eliminated_features_names)
            most_common = counter_features.most_common(num_features_to_eliminate['quantity_unimportance_features'])
            names = [name_count[0] for name_count in most_common]
            return names
        
    @staticmethod
    def _remove_conflicting_hyperparameters(hyperparameters):
        if 'bootstrap_type' not in hyperparameters:
            return hyperparameters
        
        if hyperparameters['bootstrap_type'] == 'Bayesian':
            if 'subsample' in hyperparameters:
                hyperparameters.pop('subsample')
                
        else:
            if 'bagging_temperature' in hyperparameters:
                hyperparameters.pop('bagging_temperature')
        return hyperparameters
    
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.SearchPenalties.value, cat_boost)
    def search_penalties(self, severity_of_disease, silently=True):
        operation = Enums.OperationsOfMachineLearning.SearchPenalties.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params   = self._get_saved_parameters(severity_of_disease)
        exist_penalties = loaded_params.pop('penalties')
        loaded_params   = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
        
        exist_penalties.pop('logloss', None)
        quantity_unresulting_calculations = exist_penalties.pop('quantity_unresulting_calculations', 0)
        if quantity_unresulting_calculations >= self.limit_unresulting_calculations:
            return self.get_previous_result_for_passing_calculation(operation, exist_penalties)
        
        all_increment_metric = 0
        quantity_shuffling_parameters = 0
        loaded_params['penalties'] = exist_penalties
        
        #search common_penalty
        if 'common_penalty' not in loaded_params['penalties']:
            common_penalty, increment_metric  = self._search_common_penalty(severity_of_disease, loaded_params, quantity_shuffling_parameters, silently)
            loaded_params['penalties'].update(common_penalty)
            all_increment_metric += increment_metric
        else:
            common_penalty = {'common_penalty': loaded_params['penalties']['common_penalty']}
        
        #search separated penalties 
        separated_penalties, increment_metric = self._search_separated_penalties(severity_of_disease, loaded_params, quantity_shuffling_parameters, silently)
        separated_penalties = {'separated_penalties': separated_penalties}
        loaded_params['penalties'].update(separated_penalties)
        all_increment_metric += increment_metric
        
        #search penalties_coefficient
        penalties_coefficient, metrics, increment_metric = self._search_penalty_coefficient(severity_of_disease, loaded_params, quantity_shuffling_parameters, silently)
        all_increment_metric += increment_metric
        
        if all_increment_metric < self.calculation_expediency_threshold:
            quantity_unresulting_calculations += 1 
        else:
            quantity_unresulting_calculations = 0
        
        need_to_continue_optimization = (quantity_unresulting_calculations < self.limit_unresulting_calculations)
        
        metrics       = {'logloss': metrics}
        parameters    = common_penalty | separated_penalties | penalties_coefficient | metrics | {'quantity_unresulting_calculations': quantity_unresulting_calculations}
        common_result = Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, need_to_continue_optimization)
        return common_result
        
    def _search_common_penalty(self, severity_of_disease, loaded_params, quantity_shuffling_parameters, silently=True):
        loaded_params['penalties'].pop('common_penalty', None)
        
        name_variabling_params      = 'common_penalty'
        random_numbers              = self._genearte_randoms(self.quantity_iterations_for_stable_common_penalty)
        counted_func_common_penalty = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, 
                                                        loaded_params, self.regularized_metric, name_variabling_params)
        
        searcher = GridSearch(counted_func_common_penalty, self._compare_metric, silently, quantity_shuffling_parameters, 
                              quantity_continuously_bad_steps_before_break=3)
        
        #for most features, point pow(10, -4) gives exactly the same metric value as point pow(10, -1), pow(10, -8) has almost no effect
        parameters = {}
        parameters['common_penalty'] = {'values': [pow(10, -7), pow(10, -6), pow(10, -5)]}
        
        result = searcher.count(parameters)
        return result[0], result[2]
                
    def _search_separated_penalties(self, severity_of_disease, loaded_params, quantity_shuffling_parameters, silently=True):    
        def get_penalties_parameters(min_value, max_value, multiplier):
            values = []
            current_value = min_value
            while current_value <= max_value:
                values.append(current_value)
                current_value *= multiplier
            return values
        
        def get_all_features():
            eliminated_features = loaded_params['eliminated_features']
            if eliminated_features:
                eliminated_features = eliminated_features['eliminated_features']
            else:
                eliminated_features = []
                
            learning_data   = self._create_learning_data(severity_of_disease, eliminated_features=eliminated_features)
            current_features= learning_data.get_current_features()
            return current_features.get_all_features()
        
        separated_penalties = loaded_params['penalties'].pop('separated_penalties', {})
        quantity_iterations = self._get_increasing_quantity_for_gradient_mode(self.quantity_iterations_for_stable_penalty, separated_penalties)
             
        #since there are a lot of penalties, we use few random points for the calculation,
        #therefore, for the stability of the solution, we use the metric without regularization (this will be corrected penalty_coefficient)
        metric_func             = (lambda _, test: test) 
        name_variabling_params  = 'separated_penalties'       
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, 
                                                    loaded_params, metric_func, name_variabling_params)
        parameters     = {}
        all_features   = get_all_features()
        common_penalty = loaded_params['penalties']['common_penalty']
        if separated_penalties:
            #point pow(10, -9) is equivalent to absence penalty and for most features, point pow(10, -4) gives the same metric value as point pow(10, -1)
            values_penalties = get_penalties_parameters(min_value=pow(10, -8), max_value=pow(10, -4), multiplier=5)
            for feature in all_features:
                parameters[feature] = {'values': values_penalties}
                
            #set start values
            for feature, value_penalty in separated_penalties.items():
                if feature not in parameters:
                    continue
                
                parameters[feature]['start'] = Functions.find_nearest_value_in_array(values_penalties, value_penalty)
                if parameters[feature]['start'] is None:
                    raise RuntimeError("Couldn't find the nearest point in the array while count penalties!")
            
            #'start' value may be not installed, since due to the elimination of different features on different iterations and initial value may not be in the saved penalty
            for feature in all_features:
                if 'start' in parameters[feature]:
                    continue
                
                parameters[feature]['start'] = Functions.find_nearest_value_in_array(values_penalties, 10*common_penalty)
                if parameters[feature]['start'] is None:
                    raise RuntimeError("Couldn't find the nearest point in the array while count penalties!")
            
            operation        = Enums.OperationsOfMachineLearning.SearchPenalties.value    
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state     = quantity_shuffling_parameters + number_iteration - 1
            searcher         = GradientSearch(counted_func, self._compare_metric, random_state, silently, set_maximum_parameters=True)
            
        else:
            #for most features, point pow(10, -4) gives exactly the same metric value as point pow(10, -1)
            values_penalties = get_penalties_parameters(min_value=pow(10, -7), max_value=pow(10, -5), multiplier=10) 
            for feature in all_features:
                parameters[feature] = {'values': values_penalties, 'start':common_penalty}
            
            quantity_continuously_bad_steps_before_break= 2
            quantity_all_bad_steps_before_break         = 3
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters, 
                                  quantity_continuously_bad_steps_before_break, quantity_all_bad_steps_before_break, set_maximum_parameters=True)
            
        result = searcher.count(parameters) 
        return result[0], result[2]
        
    def _search_penalty_coefficient(self, severity_of_disease, loaded_params, quantity_shuffling_parameters, silently=True):
        penalties_coefficient = loaded_params['penalties'].pop('penalties_coefficient', None)
        quantity_iterations   = self.quantity_iterations_for_stable_penalties_coefficient
        quantity_iterations   = self._get_increasing_quantity_for_gradient_mode(quantity_iterations, penalties_coefficient)
            
        name_variabling_params  = 'penalties_coefficient'       
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, 
                                                    loaded_params, self.regularized_metric, name_variabling_params)
        parameters = {}
        if penalties_coefficient:
            values_for_penalty_coefficient = [1/8, 1/4, 1/3, 1/2, 1, 2, 3, 4, 8]
            parameters['penalties_coefficient'] = {'values': values_for_penalty_coefficient, 'start': penalties_coefficient}
            
            operation        = Enums.OperationsOfMachineLearning.SearchPenalties.value    
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state     = quantity_shuffling_parameters + number_iteration - 1
            searcher         = GradientSearch(counted_func, self._compare_metric, random_state, silently, set_maximum_parameters=True)
        else:
            values_for_penalty_coefficient = [1/8, 1/4, 1/2, 1, 2, 4, 8]
            parameters['penalties_coefficient'] = {'values': values_for_penalty_coefficient, 'start': 1}
            
            quantity_continuously_bad_steps_before_break= len(values_for_penalty_coefficient)
            quantity_all_bad_steps_before_break         = len(values_for_penalty_coefficient)
            searcher                                    = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters,
                                                                     quantity_continuously_bad_steps_before_break, quantity_all_bad_steps_before_break, 
                                                                     set_maximum_parameters=True)
        result = searcher.count(parameters)    
        return result[0], result[1], result[2]
            
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.SearchHyperparameters.value, cat_boost)
    def search_hyperparameters(self, severity_of_disease,  silently=True):
        operation = Enums.OperationsOfMachineLearning.SearchHyperparameters.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params         = self._get_saved_parameters(severity_of_disease)
        exist_hyperparameters =  loaded_params.pop('hyperparameters')
        loaded_params         = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params) 
        
        exist_hyperparameters.pop('logloss', None)
        quantity_unresulting_calculations = exist_hyperparameters.pop('quantity_unresulting_calculations', 0)
        if quantity_unresulting_calculations >= self.limit_unresulting_calculations:
            return self.get_previous_result_for_passing_calculation(operation, exist_hyperparameters)
        
        quantity_iterations     = self.quantity_iterations_for_stable_parameters
        quantity_iterations     = self._get_increasing_quantity_for_gradient_mode(quantity_iterations, exist_hyperparameters)
            
        name_variabling_params  = 'hyperparameters'       
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, name_variabling_params)
        
        parameters = {}
        quantity_shuffling_parameters = 1
        parameters['bootstrap_type'] = {'values': ['Bayesian', 'Bernoulli', 'MVS'], 'slave_parameters':['bagging_temperature', 'subsample', 'subsample'], 'start':'MVS'} 
        if exist_hyperparameters:
            parameters['bagging_temperature']       = {'min_value':0,   'max_value':10, 'step':1}
            parameters['subsample']                 = {'min_value':0.6, 'max_value':1,  'step':0.05}
            parameters['one_hot_max_size']          = {'min_value':1,   'max_value':10, 'step':1}
            parameters['leaf_estimation_iterations']= {'min_value':1,   'max_value':30, 'step':1}
            parameters['max_ctr_complexity']        = {'min_value':1,   'max_value':8,  'step':1}
            parameters['random_strength']           = {'min_value':0,   'max_value':10, 'step':1}
            
            for parameter, value in exist_hyperparameters.items():
                parameters[parameter]['start'] = value
            
            if 'bagging_temperature' not in exist_hyperparameters:
                parameters['bagging_temperature']['start'] = 1 #default value
                
            if 'subsample' not in exist_hyperparameters:
                if parameters['bootstrap_type']['start'] in ['Poisson', 'Bernoulli']:
                    parameters['subsample']['start'] = 0.66 #default value
                else:
                    parameters['subsample']['start'] = 0.8 #default value
            
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state = quantity_shuffling_parameters + number_iteration - 1        
            searcher = GradientSearch(counted_func, self._compare_metric, random_state, silently)
            
        else:
            parameters['bagging_temperature']       = {'values': [1, 3, 5],         'start':1}
            parameters['subsample']                 = {'values': [0.7, 0.75, 0.8],  'start':0.8}
            parameters['one_hot_max_size']          = {'values': [1, 3, 5],         'start':1}
            parameters['leaf_estimation_iterations']= {'values': [1, 5, 10, 15, 20],'start':10}
            parameters['max_ctr_complexity']        = {'values': [2, 4, 6],         'start':4}
            parameters['random_strength']           = {'values': [1, 3, 5],         'start':1}
            
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters)
        
        result             = searcher.count(parameters)    
        optimum_parameters = result[0]
        optimum_parameters = self._remove_conflicting_hyperparameters(optimum_parameters)
        
        increment_metric = result[2]
        if increment_metric < self.calculation_expediency_threshold:
            quantity_unresulting_calculations += 1 
        else:
            quantity_unresulting_calculations = 0
        
        need_to_continue_optimization = (quantity_unresulting_calculations < self.limit_unresulting_calculations)
        
        metrics         = {'logloss': result[1]}
        parameters      = optimum_parameters | metrics | {'quantity_unresulting_calculations': quantity_unresulting_calculations}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, need_to_continue_optimization)
        return result_operation 
        
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.OptimizeTrainTestSplit.value, cat_boost) 
    def optimize_train_test_split(self, severity_of_disease, silently=True):
        operation = Enums.OperationsOfMachineLearning.OptimizeTrainTestSplit.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params    = self._get_saved_parameters(severity_of_disease)
        exist_train_size = loaded_params.pop('train_size')
        loaded_params    = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
        
        exist_train_size.pop('logloss', None)
        quantity_unresulting_calculations = exist_train_size.pop('quantity_unresulting_calculations', 0)
        if quantity_unresulting_calculations >= self.limit_unresulting_calculations:
            return self.get_previous_result_for_passing_calculation(operation, exist_train_size)
        
        
        quantity_iterations = self.quantity_iterations_for_stable_train_test_split
        quantity_iterations = self._get_increasing_quantity_for_gradient_mode(quantity_iterations, exist_train_size)
                     
        name_variabling_params = 'train_size'        
        random_numbers = self._genearte_randoms(quantity_iterations)
        counted_func = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, name_variabling_params)
         
        train_test_split = {}   
        quantity_shuffling_parameters = 0
 
        if exist_train_size:
            train_test_split['train_size'] = {'min_value':0.7, 'max_value':0.9, 'step':0.01}
            train_test_split['train_size']['start'] = exist_train_size['train_size']
            
            number_iteration = self._get_number_iteration(operation, severity_of_disease)
            random_state = quantity_shuffling_parameters + number_iteration - 1
            searcher = GradientSearch(counted_func, self._compare_metric, random_state, silently)
        
        else:
            train_test_split['train_size'] = {'min_value':0.74, 'max_value':0.86, 'step':0.03, 'start':0.8}
            searcher                       = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters)
        
        result = searcher.count(train_test_split)  
        
        increment_metric = result[2]
        if increment_metric < self.calculation_expediency_threshold:
            quantity_unresulting_calculations += 1 
        else:
            quantity_unresulting_calculations = 0
        
        need_to_continue_optimization = (quantity_unresulting_calculations < self.limit_unresulting_calculations)
        
        metrics         = {'logloss': result[1]}
        parameters      = result[0] | metrics | {'quantity_unresulting_calculations': quantity_unresulting_calculations}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics, need_to_continue_optimization)
        return result_operation
    
    @Optimizator.save_result(Enums.OperationsOfMachineLearning.OptimizeLearningRate.value, cat_boost)
    def optimize_learning_rate(self, severity_of_disease, silently=True): 
        operation = Enums.OperationsOfMachineLearning.OptimizeLearningRate.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        #always grid search
        loaded_params       = self._get_saved_parameters(severity_of_disease)
        loaded_params       = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
        quantity_iterations = self.quantity_iterations_for_stable_learning_rate
        
        name_variabling_params  = 'learning_rate'        
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.regularized_metric, name_variabling_params)
        
        parameters  = {} 
        quantity_shuffling_parameters = 0
        parameters['learning_rate'] = {'min_value':0.03, 'max_value':self.default_learning_rate,'step':0.005, 'start': self.default_learning_rate} #max equal default_learning_rate, because overfitting parameters fit with this value
        searcher    = GridSearch(counted_func, self._compare_metric, silently, quantity_shuffling_parameters, 
                                 quantity_continuously_bad_steps_before_break=4, set_maximum_parameters=True)
        
        result      = searcher.count(parameters)  
        
        parameters      = result[0]
        metrics         = {'logloss': result[1]}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
        return result_operation
       
    def count_metrics_for_all_parameters(self, severity_of_disease, learning_rate=None, allow_incomplete_bunch_of_files=False, silently=True):
        def get_metrics(model, learning_data):
            train_predictions, test_predictions = model.get_predictions()
            needed_metrics  = ['Kappa', 'MCC']
            
            Y_train         = learning_data.get_train_samples()[1] #if use pool , then got an error
            train_metrics   = self.count_metrics(train_predictions, Y_train, needed_metrics, self.multiplier_of_zero_class)
            
            Y_test      = learning_data.get_test_samples()[1] #if use pool , then got an error
            test_metrics= self.count_metrics(test_predictions, Y_test, needed_metrics, self.multiplier_of_zero_class)
            return {'learn': train_metrics, 'validation': test_metrics}
        
        def get_bunch_of_parameters_temporary_files():
            bunch_of_parameters = []
            unlooped_operations = [operations.FeaturesProfiling, operations.AllOperationsInCircle, operations.OptimizeLearningRate]
            for current_operation in operations:
                if current_operation in unlooped_operations:
                    continue
            
                parameters_temporary_file = Enums.ParametersTemporaryFiles(self.analysis_type, current_operation.value, cat_boost, severity_of_disease)
                bunch_of_parameters.append(parameters_temporary_file)
            return bunch_of_parameters
        
        def mean_metrics(metrics, index_in_values):
            mean_metrics = {}    
            for metric, arrays in metrics.items():
                mean_metrics[metric] = self.mean_metric(arrays[index_in_values])
            return mean_metrics
            
        operations = Enums.OperationsOfMachineLearning
        if not silently:
            print('operation = Count result metrics \tseverity_of_disease = ' + severity_of_disease)
        
        if allow_incomplete_bunch_of_files:
            loaded_params = self._get_saved_parameters(severity_of_disease)
            loaded_params = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
            
            parameters_avoidance_overfitting= loaded_params['parameters_avoidance_overfitting']
            hyperparameters                 = loaded_params['hyperparameters']
            eliminated_features             = loaded_params['eliminated_features'].get('eliminated_features', [])
            penalties                       = loaded_params['penalties']
            separated_penalties             = penalties.get('separated_penalties', {})
            penalties_coefficient           = penalties.get('penalties_coefficient', 1)
            common_penalty                  = penalties.get('common_penalty', self.common_penalty)
            train_size                      = loaded_params['train_size'].get('train_size', self.default_train_size)
            
        else:
            #load data only for fully completed sets of operations, so that the positive effect of iterations after an interrupted iteration is correctly calculated
            parameters_temporary_files = get_bunch_of_parameters_temporary_files()    
            loaded_params = Functions.load_objects_from_complete_bunch_of_files(parameters_temporary_files)
            loaded_params = self.clear_parameters_from_metrics_and_quantity_unresulting_calculations(loaded_params)
            
            if loaded_params:
                #either there's data for all operations of bunch, or there's nothing
                parameters_avoidance_overfitting= loaded_params[operations.AvoidanceOverfitting.value]
                hyperparameters                 = loaded_params[operations.SearchHyperparameters.value]
                eliminated_features             = loaded_params[operations.SearchEliminatedFeatures.value]['eliminated_features']
                penalties                       = loaded_params[operations.SearchPenalties.value]
                separated_penalties             = penalties['separated_penalties']
                penalties_coefficient           = penalties['penalties_coefficient']
                common_penalty                  = penalties['common_penalty']
                train_size                      = loaded_params[operations.OptimizeTrainTestSplit.value]['train_size']
                
            else:
                parameters_avoidance_overfitting= {}
                hyperparameters                 = {}
                eliminated_features             = []
                separated_penalties             = {}
                penalties_coefficient           = 1
                common_penalty                  = self.common_penalty
                train_size                      = self.default_train_size
            
        if learning_rate is None:
            learning_rate = self.default_learning_rate
        
        #first array for test- value, second- for delta of train-value and test-value, third for common- metric
        metrics = {}
        metrics['Logloss']  = ([], [], [])
        metrics['MCC']      = ([], [], [])  
        metrics['Kappa']    = ([], [], [])
        
        for random_number in self.random_numbers_for_resulting_count:
            learning_data   = self._create_learning_data(severity_of_disease, random_number, train_size, eliminated_features)
            current_features= learning_data.get_current_features()
            pools           = Pools(current_features, learning_data)
            
            params_model = parameters_avoidance_overfitting | hyperparameters | {'penalties_coefficient': penalties_coefficient} | {'learning_rate': learning_rate}
            model = self._create_model(current_features, pools, random_number, params_model, separated_penalties, common_penalty)
            
            #MCC and Kappa counting with finding the exact boundary separating the classes
            best_score_logloss   = model.get_best_score()
            best_score_MCC_Kappa = get_metrics(model, learning_data)
            
            best_score              = {}
            best_score['learn']     = best_score_logloss['learn'] | best_score_MCC_Kappa['learn']
            best_score['validation']= best_score_logloss['validation'] | best_score_MCC_Kappa['validation']
            
            for metric, arrays in metrics.items():
                train_metric= round(100*best_score['learn'][metric], 3)
                test_metric = round(100*best_score['validation'][metric], 3)
                
                arrays[0].append(test_metric)
                if metric == 'Logloss':
                    arrays[1].append(round(test_metric-train_metric, 3))
                else:
                    arrays[1].append(round(train_metric-test_metric, 3))
                
                regularized_metric = self.regularized_metric(train_metric, test_metric, metric)    
                arrays[2].append(regularized_metric)
                
        
        if not silently and len(self.random_numbers_for_resulting_count) >= 4:
            for metric, arrays in metrics.items():
                three_quantilies = statistics.quantiles(arrays[1], n=4)
                
                quartiles = [min(arrays[1])]
                quartiles.extend(three_quantilies)
                quartiles.append(max(arrays[1]))
                quartiles = [round(q, 3) for q in quartiles]
                
                interquartile_range = round(three_quantilies[2]-three_quantilies[0], 3)
                print('quartiles ' + metric + ' = \t' + str(quartiles) + '\tinterquartile range = ' + str(interquartile_range))
            
            regularized_metrics = mean_metrics(metrics, index_in_values=2)
            print('Regularized metrics = \t' + str(regularized_metrics))
            
        test_metrics = mean_metrics(metrics, index_in_values=0)
        return test_metrics
            
      
