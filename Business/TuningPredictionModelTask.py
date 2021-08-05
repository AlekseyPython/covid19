from .ATask import ATask
from Entires import Enums
import Initialization, Functions


class Task(ATask):
    def __init__(self, signal_message, analysis_type, operation, model_type, severities_of_disease, print_only_final_results):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.analysis_type = analysis_type
        self.operation = operation
        self.model_type = model_type
        self.severities_of_disease = severities_of_disease
        self.print_only_final_results = print_only_final_results
        self.percent_limit_quality = 0.3
    
    def run(self):
        icontroller_data_sourse = Initialization.icontroller_data_sourse
        source_data = icontroller_data_sourse.read_source_data(self.signal_message)
        converted_data = icontroller_data_sourse.read_converted_data(self.signal_message, ignore_empty_values=False)
        
        table_name = Enums.get_table_hdf5_by_period_pollution(Enums.PeriodsPollution.last_year)
        air_pollution = icontroller_data_sourse.read_prepared_data(self.signal_message, table_name)
        
        learning = Initialization.fabric_machine_learning.create(self.analysis_type, self.model_type, source_data, converted_data, air_pollution)
        
        self.result = []
        silently = self.print_only_final_results
        operations = Enums.OperationsOfMachineLearning
        for severity in self.severities_of_disease:
            
            if self.operation == operations.FeaturesProfiling.value:
                learning.features_profiling(severity, silently)
                continue
                
            elif self.operation == operations.AvoidanceOverfitting.value:
                result = learning.avoidance_overfitting(severity, silently)
                
            elif self.operation == operations.SearchEliminatedFeatures.value:
                result = learning.search_eliminated_features(severity, silently)
                
            elif self.operation == operations.SearchPenalties.value:
                result = learning.search_penalties(severity, silently)
                    
            elif self.operation == operations.SearchHyperparameters.value:
                result = learning.search_hyperparameters(severity, silently)
            
            elif self.operation == operations.OptimizeTrainTestSplit.value:
                result = learning.optimize_train_test_split(severity, silently)
            
            elif self.operation == operations.OptimizeLearningRate.value:
                result = learning.optimize_learning_rate(severity, silently)
                        
            elif self.operation == operations.AllOperationsInCircle.value:
                result = self._count_all_operations_in_circle(learning, severity)
            
            else:
                raise RuntimeError('Invalid operation type passed to the task!')
            
            if self.operation != operations.AllOperationsInCircle.value:
                print(self._representation_result(result))
                
            if type(result) == list:
                self.result.extend(result)
            else:
                self.result.append(result)
    
    def _count_all_operations_in_circle(self, learning, severity):
        def exist_exceeding_limit(previous_metrics, current_metrics, limit):
            for key in current_metrics.keys():
                if current_metrics[key] - previous_metrics[key] > limit:
                    return True
            return False
        
        def remove_all_element_except_first(array):
            if len(array) > 0:
                first_element = array[0]
                array.clear()
                array.append(first_element)
            return array
        
        def exist_result(valid_saved_results, current_operation):
            if current_operation in valid_saved_results:
                valid_saved_results.remove(current_operation)
                return True
            return False
                
        results_all_operations = []    
        silently = self.print_only_final_results
        
        previous_metrics = None
        current_metrics = learning.count_metrics_for_all_parameters(severity, silently=False) 
        results_all_operations.append({'Result metrics for all parameters': current_metrics})
        print('Result metrics for all parameters = ' + str(current_metrics))
        return
    
        operations = Enums.OperationsOfMachineLearning
        valid_saved_results = self._get_valid_saved_results_of_operations(severity)
        while previous_metrics is None or exist_exceeding_limit(previous_metrics, current_metrics, self.percent_limit_quality):
            #return results only last iteration
            results_all_operations = remove_all_element_except_first(results_all_operations)
            
            if not exist_result(valid_saved_results, operations.AvoidanceOverfitting.value):
                result = learning.avoidance_overfitting(severity, silently)
                results_all_operations.append(result)
                print(self._representation_result(result))
            
            if not exist_result(valid_saved_results, operations.SearchEliminatedFeatures.value):
                result = learning.search_eliminated_features(severity, silently)
                results_all_operations.append(result)
                print(self._representation_result(result))
                
            if not exist_result(valid_saved_results, operations.SearchPenalties.value):
                result = learning.search_penalties(severity, silently)
                results_all_operations.append(result)
                print(self._representation_result(result))
        
            if not exist_result(valid_saved_results, operations.SearchHyperparameters.value):
                result = learning.search_hyperparameters(severity, silently)
                results_all_operations.append(result)
                print(self._representation_result(result))
        
            if not exist_result(valid_saved_results, operations.OptimizeTrainTestSplit.value):
                result = learning.optimize_train_test_split(severity, silently)
                results_all_operations.append(result)
                print(self._representation_result(result))
        
            previous_metrics = current_metrics
            current_metrics = learning.count_metrics_for_all_parameters(severity, silently=False)  
            results_all_operations.append({'Result metrics for all parameters': current_metrics})
            print('Result metrics for all parameters = ' + str(current_metrics))
        
        #learning rate optimize for all counts
        result_learning_rate = learning.optimize_learning_rate(severity, silently)  
        results_all_operations.append(result_learning_rate)
        print(self._representation_result(result_learning_rate))
        
        learning_rate = result_learning_rate.parameters['learning_rate']
        current_metrics = learning.count_metrics_for_all_parameters(severity, learning_rate, silently=False)
        results_all_operations.append({'Result metrics for all parameters': current_metrics})
        print('Result metrics for all parameters = ' + str(current_metrics))
                     
        return results_all_operations
    
    def _get_valid_saved_results_of_operations(self, severity):  
        def get_last_file_for_operation(operation):
            parameters_temporary_file = Enums.ParametersTemporaryFiles(self.analysis_type, operation, self.model_type, severity)
            return Functions.get_last_temporary_file(parameters_temporary_file)
        
        valid_saved_results = []
        previous_file_operation = None
        for operation in Enums.OperationsOfMachineLearning:
            if operation == Enums.OperationsOfMachineLearning.FeaturesProfiling:
                continue
            
            file_operation = get_last_file_for_operation(operation.value)
            if file_operation is None:
                break
            
            if previous_file_operation is None:
                valid_saved_results.append(operation.value)
            
            else:
                if file_operation > previous_file_operation:
                    valid_saved_results.append(operation.value)
                else:
                    break
            previous_file_operation = file_operation
            
        if len(valid_saved_results) == len(Enums.OperationsOfMachineLearning):
            #all operations have valid files, start new round
            valid_saved_results.clear()
            
        return valid_saved_results  
        
    @staticmethod
    def _representation_result(result_opeartion: Enums.ResultOperationOfMachineLearning):
        name_metric = list(result_opeartion.metrics.keys())[0]
        value_metric = result_opeartion.metrics[name_metric]
        return 'Operation = ' + str(result_opeartion.operation) + '\t' + name_metric + ' = ' + str(value_metric)    
                
                
                
            
            
            
        
        
                       
        