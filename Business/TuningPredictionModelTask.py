from .ATask import ATask
from Entires import Enums
import Initialization, Functions


class Task(ATask):
    def __init__(self, signal_message, analysis_type, operation, severities_of_disease, print_only_final_results):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.analysis_type = analysis_type
        self.operation = operation
        self.severities_of_disease = severities_of_disease
        self.print_only_final_results = print_only_final_results
        self.calculation_expediency_threshold = 0.1
        self.limit_unresulting_calculations = 1
    
    def run(self):
        optimizator = self._get_optimizator()
        
        self.result = []
        silently = self.print_only_final_results
        operations = Enums.OperationsOfMachineLearning
        for severity in self.severities_of_disease:
            
            if self.operation == operations.NeuralNetworkTraining.value:
                result = optimizator.learn(severity, silently=False)
                continue
                
            elif self.operation == operations.FeaturesProfiling.value:
                optimizator.features_profiling(severity, silently)
                continue
                
            elif self.operation == operations.AvoidanceOverfitting.value:
                result = optimizator.avoidance_overfitting(severity, silently)
                
            elif self.operation == operations.SearchEliminatedFeatures.value:
                result = optimizator.search_eliminated_features(severity, silently)
                
            elif self.operation == operations.SearchPenalties.value:
                result = optimizator.search_penalties(severity, silently)
                    
            elif self.operation == operations.SearchHyperparameters.value:
                result = optimizator.search_hyperparameters(severity, silently)
            
            elif self.operation == operations.OptimizeTrainTestSplit.value:
                result = optimizator.optimize_train_test_split(severity, silently)
            
            elif self.operation == operations.OptimizeLearningRate.value:
                result = optimizator.optimize_learning_rate(severity, silently)
                        
            elif self.operation == operations.AllOperationsInCircle.value:
                result = self._count_all_operations_in_circle(optimizator, severity)
            
            else:
                raise RuntimeError('Invalid operation type passed to the task!')
            
            if self.operation != operations.AllOperationsInCircle.value:
                print(self._representation_result(result))
                
            if type(result) == list:
                self.result.extend(result)
            else:
                self.result.append(result)
    
    def _get_optimizator(self):
        icontroller_data_sourse = Initialization.icontroller_data_sourse
        source_data = icontroller_data_sourse.read_source_data(self.signal_message)
        converted_data = icontroller_data_sourse.read_converted_data(self.signal_message, ignore_empty_values=False)
        
        if self.operation == Enums.OperationsOfMachineLearning.NeuralNetworkTraining.value:
            return Initialization.fabric_neural_network.create_optimizator(source_data, converted_data)
        
        else:
            table_name = Enums.get_table_hdf5_by_period_pollution(Enums.PeriodsPollution.last_year)
            air_pollution = icontroller_data_sourse.read_prepared_data(self.signal_message, table_name)
            
            model_type = Enums.TypesOfGradientBoostingModel.CatBoost.value
            return Initialization.fabric_gradient_boosting.create_optimizator(self.analysis_type, model_type, source_data, converted_data, 
                                                                              air_pollution, self.calculation_expediency_threshold, 
                                                                              self.limit_unresulting_calculations)

    def _count_all_operations_in_circle(self, optimizator, severity):
        results_all_operations = []    
        silently = self.print_only_final_results
        
        current_metrics = optimizator.count_metrics_for_all_parameters(severity, silently=False) 
        results_all_operations.append({'Result metrics for all parameters': current_metrics})
        print('Result test- metrics for all parameters = ' + str(current_metrics))
        
        operations = Enums.OperationsOfMachineLearning
        
        optimizators = []
        optimizators.append((operations.AvoidanceOverfitting.value,     optimizator.avoidance_overfitting))
        optimizators.append((operations.SearchEliminatedFeatures.value, optimizator.search_eliminated_features))
        optimizators.append((operations.SearchPenalties.value,          optimizator.search_penalties))
        optimizators.append((operations.SearchHyperparameters.value,    optimizator.search_hyperparameters))
        optimizators.append((operations.OptimizeTrainTestSplit.value,   optimizator.optimize_train_test_split))
            
        need_to_continue_optimization = True
        valid_saved_results = self._get_valid_saved_results_of_operations(severity)
        while need_to_continue_optimization:
        
            need_to_continue_optimization = False
            for name_operation, operation in optimizators:
                if name_operation in valid_saved_results:
                    valid_saved_results.remove(name_operation)
                    optimizator.jump_operation(name_operation, severity)
                    continue
                
                result = operation(severity, silently)
                results_all_operations.append(result)
                need_to_continue_optimization = need_to_continue_optimization or result.need_to_continue_optimization 
                print(self._representation_result(result))
                
        #learning rate optimize for all counts
        result_learning_rate = optimizator.optimize_learning_rate(severity, silently)  
        results_all_operations.append(result_learning_rate)
        print(self._representation_result(result_learning_rate))
        
        learning_rate = result_learning_rate.parameters['learning_rate']
        current_metrics = optimizator.count_metrics_for_all_parameters(severity, learning_rate, silently=False)
        results_all_operations.append({'Result metrics for all parameters': current_metrics})
        print('Result test- metrics for all parameters = ' + str(current_metrics))
                     
        return results_all_operations
    
    def _get_valid_saved_results_of_operations(self, severity):  
        def get_last_file_for_operation(operation):
            model_type = Enums.TypesOfGradientBoostingModel.CatBoost.value
            parameters_temporary_file = Enums.ParametersTemporaryFiles(self.analysis_type, operation, model_type, severity)
            return Functions.get_last_temporary_file(parameters_temporary_file)
        
        valid_saved_results = []
        previous_file_operation = None
        operations = Enums.OperationsOfMachineLearning
        unlooped_operations = [operations.FeaturesProfiling, operations.AllOperationsInCircle, operations.OptimizeLearningRate]
        
        for operation in operations:
            if operation in unlooped_operations:
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
            
        if len(valid_saved_results) == len(Enums.OperationsOfMachineLearning) - len(unlooped_operations):
            #all operations have valid files, start new round
            valid_saved_results.clear()
            
        return valid_saved_results  
        
    @staticmethod
    def _representation_result(result_opeartion: Enums.ResultOperationOfMachineLearning):
        name_metric = list(result_opeartion.metrics.keys())[0]
        value_metric = result_opeartion.metrics[name_metric]
        return 'Operation = ' + str(result_opeartion.operation) + '\t' + name_metric + ' = ' + str(value_metric)
        