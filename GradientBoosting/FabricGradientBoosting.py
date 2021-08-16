from Business.AMachineLearning import AFabricGradientBoosting
from GradientBoosting.CatBoostOptimizator import CatBoostOptimizator
from GradientBoosting import XGBoostOptimizator


class FabricGradientBoosting(AFabricGradientBoosting):
    def __init__(self):
        AFabricGradientBoosting.__init__(self)
    
    def create_optimizator(self, analysis_type, model_type, source_data, converted_data, air_pollution, calculation_expediency_threshold, limit_unresulting_calculations):
        if model_type == 'CatBoost':
            return CatBoostOptimizator(analysis_type, source_data, converted_data, air_pollution, calculation_expediency_threshold, limit_unresulting_calculations)
        
        elif model_type == 'XGBoost':
            return XGBoostOptimizator(analysis_type, source_data, converted_data, air_pollution, calculation_expediency_threshold, limit_unresulting_calculations)
        
        else:
            raise RuntimeError('Invalid model type passed to the task!')
        