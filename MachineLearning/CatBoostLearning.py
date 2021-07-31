import functools
import statistics
import numpy as np
from collections import Counter
from functools import lru_cache
from Business.AMachineLearning import AMachineLearning
from .Learning import Learning
from .LearningData import Features, PreparedData, LearningData
from .CatBoostModel import Pools, CatBoostModel
from Entires import Enums
from Entires.GridSearch import GridSearch
from Entires.GradientSearch import GradientSearch
import Settings, Functions

CatBoost = Enums.TypesOfMachineLearningModel.CatBoost.value

class CatBoostLearning(AMachineLearning, Learning):
    def __init__(self, analysis_type, source_data, converted_data, air_pollution):
        AMachineLearning.__init__(self)
        Learning.__init__(self)
        
        self.analysis_type = analysis_type
        
        np.random.seed(42)
        self.common_penalty            = pow(10, -8)
        self.default_train_size        = 0.8
        self.default_learning_rate     = 0.05
        
        #tolerance coefficient overfitting, when the result is improved. an increase in this coefficient leads to an increase in randomness,
        #and the parameters of model may be chosen incorrectly
        ratio_of_importance_optimization_to_overfitting = 1
        
        #regularize by the overfitting
        self.common_metric = (lambda train, test: test + ratio_of_importance_optimization_to_overfitting * (test-train))
        
        if Settings.debuge_mode:
            self.quantity_iterations_for_stable_parameters                  = 1
            self.quantity_iterations_for_stable_eliminated_features         = 1
            self.quantity_iterations_for_stable_penalty                     = 1
            self.quantity_iterations_for_stable_common_penalties_coefficient= 1
            self.quantity_iterations_for_stable_train_test_split            = 1
            self.quantity_iterations_for_resulting_count                    = 1
            self.quantity_iterations_for_stable_learning_rate               = 1
            
        else:
            self.quantity_iterations_for_stable_parameters                  = 25 
            self.quantity_iterations_for_stable_eliminated_features         = 10
            self.quantity_iterations_for_stable_penalty                     = 5
            self.quantity_iterations_for_stable_common_penalties_coefficient= 25
            self.quantity_iterations_for_stable_train_test_split            = 25
            self.quantity_iterations_for_resulting_count                    = 25
            self.quantity_iterations_for_stable_learning_rate               = 25
        
        self.increasing_percent_iterations_for_gradation_mode = 30
        self.random_numbers_for_resulting_count = self._genearte_randoms(self.quantity_iterations_for_resulting_count)
        
        self.features = Features(analysis_type, air_pollution)
        self.prepared_data = PreparedData(analysis_type, self.features, source_data, converted_data, air_pollution)
    
    @staticmethod
    def _compare_metric(first, second):
        #use logloss (cross entropy), for which the less the better
        if second < first:
            return 1
        elif second > first:
            return -1
        else:
            return 0
        
    def _create_learning_data(self, severity_of_disease, random_state=42, train_size=None, eliminated_features=[]):
        if train_size is None:
            train_size = self.default_train_size
            
        return LearningData(self.prepared_data, self.features, severity_of_disease, random_state, train_size, eliminated_features)
    
    def _create_model(self, current_features, pools, random_state, params_model={}, penalties={}, common_penalty=None, fit_model=True):
        if common_penalty is None:
            common_penalty = self.common_penalty
            
        model = CatBoostModel(current_features, pools, random_state, params_model, penalties, common_penalty)
        if fit_model: model.fit()
        return model
    
    def _genearte_randoms(self, quantity):
        if Settings.debuge_mode:
            np.random.seed(42)
            
        return np.random.randint(low=0, high=10000, size=quantity)
    
    @lru_cache 
    def _get_pool(self, severity_of_disease, random_number, train_size, eliminated_features_str):
        eliminated_features = eliminated_features_str.split()  
        learning_data       = self._create_learning_data(severity_of_disease, random_number, train_size, eliminated_features)
        current_features    = learning_data.get_current_features()
        pools               = Pools(current_features, learning_data)
        return current_features, pools
        
    def _get_saved_parameters(self, severity_of_disease):
        all_params = {}
        operations = Enums.OperationsOfMachineLearning
        cat_boost = Enums.TypesOfMachineLearningModel.CatBoost.value
        
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
    
    @Learning.save_result(Enums.OperationsOfMachineLearning.AvoidanceOverfitting.value, CatBoost)  
    def avoidance_overfitting(self, severity_of_disease, silently=True):
        operation = Enums.OperationsOfMachineLearning.AvoidanceOverfitting.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params = self._get_saved_parameters(severity_of_disease)
        exist_parameters = loaded_params.pop('parameters_avoidance_overfitting') 
        quantity_iterations = self.quantity_iterations_for_stable_parameters
        if exist_parameters:
            quantity_iterations *= (1 + self.increasing_percent_iterations_for_gradation_mode/100)
            quantity_iterations = round(quantity_iterations)
        
        name_variabling_params = 'parameters_avoidance_overfitting'
        random_numbers = self._genearte_randoms(quantity_iterations)
        counted_func = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params)
        
        parameters = {}
        if exist_parameters:
            parameters['early_stopping_rounds'] = {'min_value':3,       'max_value':30, 'step':3}
            parameters['l2_leaf_reg']           = {'min_value':0,       'max_value':10, 'step':1, 'set_maximum_parameter':True}
            parameters['depth']                 = {'min_value':1,       'max_value':10, 'step':1}
            parameters['boosting_type']         = {'values':['Ordered', 'Plain']}
        
            for parameter, value in exist_parameters.items():
                parameters[parameter]['start'] = value
            
            searcher = GradientSearch(counted_func, self._compare_metric, silently)
    
        else:
            parameters['early_stopping_rounds'] = {'min_value':5,       'max_value':20, 'step':5,   'start':5}
            parameters['l2_leaf_reg']           = {'min_value':2,       'max_value':8,  'step':2,   'start':8, 'set_maximum_parameter':True}
            parameters['depth']                 = {'min_value':3,       'max_value':8,  'step':2,   'start':5}
            parameters['boosting_type']         = {'values':['Ordered', 'Plain'], 'start':'Ordered'}
            
            searcher = GridSearch(counted_func, self._compare_metric, silently)
        
        result = searcher.count(parameters)
        
        parameters      = result[0]
        metrics         = {'logloss':result[1]}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
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
        
        elif name_variabling_params == 'penalties':
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
        if common_penalty is None:
            if loaded_params.get('penalties', False):
                common_penalty = loaded_params['penalties']['common_penalty']
            else:
                common_penalty = self.common_penalty
                
        if separated_penalties is None:
            if loaded_params.get('penalties', False):
                separated_penalties = loaded_params['penalties']['separated_penalties']
            else:
                separated_penalties = {}
                
        if penalties_coefficient is None:
            if loaded_params.get('penalties', False):
                common_coefficient = loaded_params['penalties']['penalties_coefficient']
            else:
                common_coefficient = 1
            penalties_coefficient = {'penalties_coefficient': common_coefficient}
        
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
        all_unimportance_features = Counter()
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
            print('logloss=' + str(metric))
            
        if name_variabling_params == 'quantity_unimportance_features':
            allowed_number_of_hits = len(random_numbers) / 10
            get_eliminated_features = (lambda border, sign=1: [key_value[0] for key_value in all_unimportance_features.items() if sign*key_value[1]>border])
            
            if allowed_number_of_hits >= 1:
                allowed_number_of_hits = round(allowed_number_of_hits)
                eliminated_features = get_eliminated_features(allowed_number_of_hits)
                quantity_unimportance_features = len(eliminated_features)
            
            else:
                fully_eliminated_features = get_eliminated_features(1)
                parted_eliminated_features = get_eliminated_features(-2, -1)
                quantity_unimportance_features = len(fully_eliminated_features) + round(len(parted_eliminated_features) * (1 - allowed_number_of_hits))
                
            return quantity_unimportance_features 
            
        return metric
    
    @Learning.save_result(Enums.OperationsOfMachineLearning.SearchEliminatedFeatures.value, CatBoost)                      
    def search_eliminated_features(self, severity_of_disease, silently=True):
        def get_eliminated_features_parameters():
            silently                       = True
            name_variabling_params         = 'quantity_unimportance_features'
            quantity_unimportance_features = self._count_model_for_point(severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params, variabling_params={})
            quantity_all_features          = len(self.features.get_all_features())
            
            parameters = {}
            parameters['quantity_unimportance_features'] = {'min_value':quantity_unimportance_features, 'max_value':quantity_all_features, 'step':1}
            return parameters
        
        operation = Enums.OperationsOfMachineLearning.SearchEliminatedFeatures.value    
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        #we use few random points for the calculation, because it is a very long procedure 
        #therefore, for the stability of the solution, we use the metric without regularization
        metric_func               = (lambda _, test: test) 
        loaded_params             = self._get_saved_parameters(severity_of_disease)
        exist_eliminated_features = loaded_params.pop('eliminated_features') 
        quantity_iterations       = self.quantity_iterations_for_stable_eliminated_features
        if exist_eliminated_features:
            quantity_iterations *= (1 + self.increasing_percent_iterations_for_gradation_mode/100)
            quantity_iterations = round(quantity_iterations)
        
        return_metrics          = True
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func = functools.partial(self._search_eliminated_features_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, return_metrics)
        
        eliminated_features = get_eliminated_features_parameters()
        if exist_eliminated_features:
            eliminated_features['quantity_unimportance_features']['start'] = len(exist_eliminated_features['eliminated_features'])
            searcher = GradientSearch(counted_func, self._compare_metric, silently, set_maximum_parameters=True)
        else:
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_continuously_bad_steps_before_break=3, set_maximum_parameters=True)
        
        result = searcher.count(eliminated_features)
        metrics = result[1]
        
        #get names eliminated features instead quantity
        return_metrics            = False
        num_features_to_eliminate = result[0]
        eliminated_features_names = self._search_eliminated_features_for_point(severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, return_metrics, num_features_to_eliminate)
        
        parameters      = {'eliminated_features': eliminated_features_names}
        metrics         = {'logloss': metrics}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
        
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
            #eliminated_features_str = ' '.join([])
            # current_features, pools = self._get_pool(severity_of_disease, random_number, train_size, '')
            
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
                print('logloss=' + str(metric))
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
    
    @Learning.save_result(Enums.OperationsOfMachineLearning.SearchPenalties.value, CatBoost) 
    def search_penalties(self, severity_of_disease, silently=True):
        def get_penalties_parameters(min_value, max_value, multiplier):
            values = []
            current_value = min_value
            while current_value <= max_value:
                values.append(current_value)
                current_value *= multiplier
            return values
        
        operation = Enums.OperationsOfMachineLearning.SearchPenalties.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        #First, find separated penalties
        loaded_params       = self._get_saved_parameters(severity_of_disease)
        exist_penalties     = loaded_params.pop('penalties') 
        quantity_iterations = self.quantity_iterations_for_stable_penalty
        if exist_penalties:
            quantity_iterations *= (1 + self.increasing_percent_iterations_for_gradation_mode/100)
            quantity_iterations = round(quantity_iterations)
             
        #since there are a lot of penalties, we use few random points for the calculation,
        #therefore, for the stability of the solution, we use the metric without regularization (this will be corrected common_penalties_coefficient)
        metric_func             = (lambda _, test: test) 
        name_variabling_params  = 'penalties'       
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, metric_func, name_variabling_params)
        
        #get all features
        if loaded_params.get('eliminated_features', False):
            eliminated_features = loaded_params['eliminated_features']['eliminated_features']
        else:
            eliminated_features = []
            
        learning_data   = self._create_learning_data(severity_of_disease, eliminated_features=eliminated_features)
        current_features= learning_data.get_current_features()
        all_features    = current_features.get_all_features()
            
        penalties = {}
        if exist_penalties:
            #point pow(10, -9) is equivalent to absence penalty and for most features, point pow(10, -4) gives the same metric value as point pow(10, -1)
            values_penalties = get_penalties_parameters(min_value=pow(10, -8), max_value=pow(10, -4), multiplier=5)
            for feature in all_features:
                penalties[feature] = {'values': values_penalties}
                
            #set start values
            common_penalties_coefficient = exist_penalties['penalties_coefficient']
            separated_penalties = exist_penalties['separated_penalties']
            for feature, value_penalty in separated_penalties.items():
                if feature not in penalties:
                    continue
                
                penalties[feature]['start'] = Functions.find_nearest_value_in_array(values_penalties, value_penalty*common_penalties_coefficient)
                if penalties[feature]['start'] is None:
                    raise RuntimeError("Couldn't find the nearest point in the array while count penalties!")
            
            #'start' value may be not installed, since due to the elimination of different features on different iterations and initial value may not be in the saved penalty
            common_penalty = {'common_penalty': exist_penalties['common_penalty']}
            for feature in all_features:
                if 'start' not in penalties[feature]:
                    penalties[feature]['start'] = 10 * common_penalty['common_penalty']
                
            searcher = GradientSearch(counted_func, self._compare_metric, silently, set_maximum_parameters=True)
            
        else:
            #search common penalties
            name_variabling_params      = 'common_penalty'
            random_numbers              = self._genearte_randoms(self.quantity_iterations_for_stable_common_penalties_coefficient)
            counted_func_common_penalty = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, metric_func, name_variabling_params)
            searcher                    = GridSearch(counted_func_common_penalty, self._compare_metric, silently, quantity_continuously_bad_steps_before_break=3, quantity_shuffling_parameters=0)
            
            #for most features, point pow(10, -4) gives exactly the same metric value as point pow(10, -1)
            values_penalties = get_penalties_parameters(min_value=pow(10, -7), max_value=pow(10, -5), multiplier=10) 
            common_penalties = {'common_penalty': {'values': values_penalties}}
            result           = searcher.count(common_penalties)
            common_penalty   = result[0]
            
            #for most features, point pow(10, -4) gives exactly the same metric value as point pow(10, -1)
            values_penalties = get_penalties_parameters(min_value=pow(10, -7), max_value=pow(10, -5), multiplier=10) 
            for feature in all_features:
                penalties[feature] = {'values': values_penalties, 'start':common_penalty['common_penalty']}
            
            quantity_continuously_bad_steps_before_break= 2
            quantity_all_bad_steps_before_break         = 3
            quantity_shuffling_parameters               = 0 #without shuffling because setted common penalties
            searcher = GridSearch(counted_func, self._compare_metric, silently, quantity_continuously_bad_steps_before_break, quantity_all_bad_steps_before_break, quantity_shuffling_parameters, set_maximum_parameters=True)
            
        result_with_separated_penalties = searcher.count(penalties) #Error !!!!!!!!!!!!!!
        current_penalties               = result_with_separated_penalties[0]
        
        #Second, find penalties_coefficient after find separated penalties (always grid, because this is an adjustment to the previous calculation). Use common_metric to correct previous calculation
        loaded_params['penalties'] = {}
        loaded_params['penalties']['separated_penalties'] = current_penalties
        loaded_params['penalties'].update(common_penalty)
        
        if exist_penalties:
            values_for_common_penalties_coefficient = [1/2, 1, 2, 4, 8]
        else:
            values_for_common_penalties_coefficient = [1/3, 1, 3, 7, 10, 30]
        
        name_variabling_params  = 'penalties_coefficient'       
        random_numbers          = self._genearte_randoms(self.quantity_iterations_for_stable_common_penalties_coefficient)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params)
        
        #count all points
        quantity_continuously_bad_steps_before_break= len(values_for_common_penalties_coefficient)
        quantity_all_bad_steps_before_break         = len(values_for_common_penalties_coefficient)
        searcher                                    = GridSearch(counted_func, self._compare_metric, silently, quantity_continuously_bad_steps_before_break, quantity_all_bad_steps_before_break, set_maximum_parameters=True)
        
        common_penalties_coefficient = {'penalties_coefficient': {'values':values_for_common_penalties_coefficient}}
        result_common_penalties_coefficient = searcher.count(common_penalties_coefficient)    
        common_penalties_coefficient = result_common_penalties_coefficient[0]
        
        all_penalties_parameters= loaded_params['penalties'] | common_penalties_coefficient | common_penalty
        all_penalties_result    = result_common_penalties_coefficient[1]
        
        parameters      = all_penalties_parameters
        metrics         = {'logloss': all_penalties_result}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
        return result_operation
            
    @Learning.save_result(Enums.OperationsOfMachineLearning.SearchHyperparameters.value, CatBoost)
    def search_hyperparameters(self, severity_of_disease,  silently=True):
        operation = Enums.OperationsOfMachineLearning.SearchHyperparameters.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params           = self._get_saved_parameters(severity_of_disease)
        exist_hyperparameters   =  loaded_params.pop('hyperparameters') 
        quantity_iterations     = self.quantity_iterations_for_stable_parameters
        if exist_hyperparameters:
            quantity_iterations *= (1 + self.increasing_percent_iterations_for_gradation_mode/100)
            quantity_iterations = round(quantity_iterations)
            
        name_variabling_params  = 'hyperparameters'       
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params)
        
        parameters = {}
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
                    
            searcher = GradientSearch(counted_func, self._compare_metric, silently)
            
        else:
            parameters['bagging_temperature']       = {'values': [1, 3, 5],         'start':1}
            parameters['subsample']                 = {'values': [0.7, 0.75, 0.8],  'start':0.8}
            parameters['one_hot_max_size']          = {'values': [1, 3, 5],         'start':1}
            parameters['leaf_estimation_iterations']= {'values': [1, 5, 10, 15, 20],'start':10}
            parameters['max_ctr_complexity']        = {'values': [2, 4, 6],         'start':4}
            parameters['random_strength']           = {'values': [1, 3, 5],         'start':1}
            
            searcher = GridSearch(counted_func, self._compare_metric, silently)
        
        result             = searcher.count(parameters)    
        optimum_parameters = result[0]
        optimum_parameters = self._remove_conflicting_hyperparameters(optimum_parameters)
        
        parameters      = optimum_parameters
        metrics         = {'logloss': result[1]}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
        return result_operation 
        
    @Learning.save_result(Enums.OperationsOfMachineLearning.OptimizeTrainTestSplit.value, CatBoost) 
    def optimize_train_test_split(self, severity_of_disease, silently=True):
        operation = Enums.OperationsOfMachineLearning.OptimizeTrainTestSplit.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        loaded_params       = self._get_saved_parameters(severity_of_disease)
        exist_train_size    = loaded_params.pop('train_size')
        quantity_iterations = self.quantity_iterations_for_stable_train_test_split
        if exist_train_size:
            quantity_iterations *= (1 + self.increasing_percent_iterations_for_gradation_mode/100)
            quantity_iterations = round(quantity_iterations)
                     
        name_variabling_params = 'train_size'        
        random_numbers = self._genearte_randoms(quantity_iterations)
        counted_func = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params)
         
        train_test_split = {}     
        if exist_train_size:
            train_test_split['train_size'] = {'min_value':0.7, 'max_value':0.9, 'step':0.01}
            train_test_split['train_size']['start'] = exist_train_size['train_size']
            searcher = GradientSearch(counted_func, self._compare_metric, silently)
        
        else:
            train_test_split['train_size'] = {'min_value':0.74, 'max_value':0.86, 'step':0.03, 'start':0.8}
            searcher                       = GridSearch(counted_func, self._compare_metric, silently)
        
        result = searcher.count(train_test_split)  
        
        parameters      = result[0]
        metrics         = {'logloss': result[1]}
        result_operation= Enums.ResultOperationOfMachineLearning(operation, parameters, metrics)
        return result_operation
    
    @Learning.save_result(Enums.OperationsOfMachineLearning.OptimizeLearningRate.value, CatBoost)
    def optimize_learning_rate(self, severity_of_disease, silently=True): 
        operation = Enums.OperationsOfMachineLearning.OptimizeLearningRate.value
        if not silently:
            print('operation = ' + str(operation) + '\tseverity_of_disease = ' + severity_of_disease)
        
        #always grid search
        loaded_params       = self._get_saved_parameters(severity_of_disease)
        quantity_iterations = self.quantity_iterations_for_stable_learning_rate
                     
        name_variabling_params  = 'learning_rate'        
        random_numbers          = self._genearte_randoms(quantity_iterations)
        counted_func            = functools.partial(self._count_model_for_point, severity_of_disease, random_numbers, silently, loaded_params, self.common_metric, name_variabling_params)
        
        parameters  = {} 
        parameters['learning_rate'] = {'min_value':0.03, 'max_value':self.default_learning_rate,'step':0.005} #max equal default_learning_rate, because overfitting parameters fit with this value
        searcher    = GridSearch(counted_func, self._compare_metric, silently, quantity_continuously_bad_steps_before_break=4, set_maximum_parameters=True)
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
            train_metrics   = self.count_metrics(train_predictions, Y_train, needed_metrics, silently=True)
            
            Y_test      = learning_data.get_test_samples()[1] #if use pool , then got an error
            test_metrics= self.count_metrics(test_predictions, Y_test, needed_metrics, silently=True)
            return train_metrics, test_metrics
        
        def get_bunch_of_parameters_temporary_files():
            bunch_of_parameters = []
            cat_boost = Enums.TypesOfMachineLearningModel.CatBoost.value
            for current_operation in operations:
                parameters_temporary_file = Enums.ParametersTemporaryFiles(self.analysis_type, current_operation.value, cat_boost, severity_of_disease)
                bunch_of_parameters.append(parameters_temporary_file)
            return bunch_of_parameters
            
        operations = Enums.OperationsOfMachineLearning
        if not silently:
            print('operation = Count result metrics \tseverity_of_disease = ' + severity_of_disease)
        
        if allow_incomplete_bunch_of_files:
            loaded_params                   = self._get_saved_parameters(severity_of_disease)
            
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
            
        test_logloss= []
        test_mcc    = []
        test_kappa  = []
        delta_logloss=[]
        delta_mcc   = []
        delta_kappa = []
        
        for random_number in self.random_numbers_for_resulting_count:
            learning_data   = self._create_learning_data(severity_of_disease, random_number, train_size, eliminated_features)
            current_features= learning_data.get_current_features()
            pools           = Pools(current_features, learning_data)
            
            params_model = parameters_avoidance_overfitting | hyperparameters | {'penalties_coefficient': penalties_coefficient} | {'learning_rate': learning_rate}
            model = self._create_model(current_features, pools, random_number, params_model, separated_penalties, common_penalty)
            
            best_score  = model.get_best_score()
            train_metric= round(100*best_score['learn']['Logloss'], 3)
            test_metric = round(100*best_score['validation']['Logloss'], 3)
            test_logloss.append(test_metric)
            delta_logloss.append(test_metric-train_metric)
            
            train_metric, test_metric = get_metrics(model, learning_data)
            test_mcc.append(test_metric['MCC'])
            test_kappa.append(test_metric['Kappa'])
            delta_mcc.append(train_metric['MCC'] - test_metric['MCC'])
            delta_kappa.append(train_metric['Kappa'] - test_metric['Kappa'])
        
        test_logloss= self.mean_metric(test_logloss)
        test_mcc    = self.mean_metric(test_mcc)
        test_kappa  = self.mean_metric(test_kappa)
            
        if not silently and len(self.random_numbers_for_resulting_count) >= 4:
            for name, delta_array in [('Logloss', delta_logloss), ('MCC', delta_mcc), ('Kappa', delta_kappa)]:
                quartiles = [min(delta_array)]
                quartiles.extend(statistics.quantiles(delta_array, n=4))
                quartiles.append(max(delta_array))
                quartiles = [round(q, 3) for q in quartiles]
                print('quartiles ' + name + ' = ' + str(quartiles))
        
        return {'Logloss':test_logloss, 'MCC': test_mcc, 'Kappa': test_kappa}
            
      
