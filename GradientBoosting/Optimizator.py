import numpy as np
import pandas as pd
from Entires import Enums
import Functions


class Optimizator:
    def __init__(self): pass
    
    def get_saved_result(self, operation, type_machine_learning_model, severity_of_disease):
        parameters_temporary_file = Enums.ParametersTemporaryFiles(self.analysis_type, operation, type_machine_learning_model, severity_of_disease)
        return Functions.load_object_from_file(parameters_temporary_file)
    
    @staticmethod
    def save_result(operation, type_machine_learning_model):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
    
                self_obj = args[0]
                severity_of_disease = args[1]
                
                saved_data = result.parameters
                parameters_temporary_file = Enums.ParametersTemporaryFiles(self_obj.analysis_type, operation, type_machine_learning_model, severity_of_disease)
                Functions.save_object_to_file(saved_data, parameters_temporary_file)
                return result
            return wrapper
        return decorator

    def mean_metric(self, metrics): 
        metrics = np.array(metrics)
        metrics = metrics[~np.isnan(metrics)]
        
        if len(metrics) >= 5:
            #remove edging points
            metrics.sort()
            
            if len(metrics) >= 20:
                size_edge = len(metrics) // 5
                
            elif len(metrics) >= 15:
                size_edge = 2
                
            else:
                size_edge = 1
                
            metrics = metrics[size_edge: -size_edge]
                
        return round(metrics.mean(), 3)
    
    def count_metrics(self, predictions, targets, needed_metrics, multiplier_of_zero_class=1):
        def get_consistent_number_intervals():  
            optimun_quantity_intervals = None
            for quantity_intervals in range(2, 100):
                size_interval = 1 / quantity_intervals
                df['interval'] = df['predictions'] // size_interval 
            
                grouped = df['targets'].groupby(df['interval'])
                means = grouped.mean()
                
                exist_violation = False
                previous_target = 0
                for interval in means.index:
                    target = means.loc[interval]
                    if target <= previous_target:
                        exist_violation = True
                        break
                    previous_target = target
                
                if exist_violation:
                    optimun_quantity_intervals = quantity_intervals-1
                    break
                
            if optimun_quantity_intervals is None:
                raise RuntimeError('It is impossible to find the optimal number of intervals for these predictions!')
            return optimun_quantity_intervals
    
        df = pd.DataFrame({'predictions':predictions, 'targets':targets})
        
        optimun_quantity_intervals = get_consistent_number_intervals()
        size_interval = 1 / optimun_quantity_intervals
        df['interval'] = df['predictions'] // size_interval
        
        grouped = df['targets'].groupby(df['interval'])
        means = grouped.mean()
        
        middle_probability = multiplier_of_zero_class / (multiplier_of_zero_class + 1)
        left_series = means[means<=middle_probability]
        right_series = means[means>middle_probability]
        
        if len(left_series)==0 or len(right_series)==0:
            middle_point = middle_probability
        else:
            min_interval = left_series.index[-1]
            min_point = min_interval * size_interval 
            min_probability = means[min_interval]
            
            max_interval = right_series.index[0]
            max_point = max_interval * size_interval
            max_probability = means[max_interval]
            
            derivative = (max_probability-min_probability) / (max_point-min_point)
            
            if middle_probability - min_probability < max_probability - middle_probability:
                middle_point = min_point + (middle_probability-min_probability) / derivative
            else:
                middle_point = max_point - (max_probability-middle_probability) / derivative
                
        TP = df[(df.predictions>middle_point) & (df.targets==True)].shape[0]
        TN = df[(df.predictions<=middle_point) & (df.targets==False)].shape[0]
        FP = df[(df.predictions>middle_point) & (df.targets==False)].shape[0]
        FN = df[(df.predictions<=middle_point) & (df.targets==True)].shape[0]
        All = TP+TN+FP+FN
        
        counted_metrics = {}
        if 'Accuracy' in needed_metrics:
            Accuracy = (TP + TN) / All
            counted_metrics['Accuracy'] = round(Accuracy, 5)
        
        if 'Balanced accuracy' in needed_metrics:
            if (TP+FN)==0 or (TN+FP)==0:
                Balanced_Accuracy = 0
            else:
                Balanced_Accuracy = 0.5 * (TP/(TP+FN) + TN/(TN+FP))
                Balanced_Accuracy = round(Balanced_Accuracy, 5)
            counted_metrics['Balanced accuracy'] = Balanced_Accuracy    
        
        if 'Kappa' in needed_metrics:
            Accuracy = (TP+TN) / All
            Accuracy_chance = (TN+FP)*(TN+FN)/(All*All) + (TP+FN)*(TP+FP)/(All*All)
            Kappa = (Accuracy-Accuracy_chance) / (1 - Accuracy_chance)
            counted_metrics['Kappa'] = round(Kappa, 5)    
                
        if 'MCC' in needed_metrics:
            if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0:
                MCC = np.nan
            else:
                MCC = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                MCC = round(MCC, 5)
            counted_metrics['MCC'] = MCC    
                
        return counted_metrics
