from Business.AMachineLearning import AFabricMachineLearning
from .CatBoostLearning import CatBoostLearning
from .XGBoostLearning import XGBoostLearning

class FabricMachineLearning(AFabricMachineLearning):
    def __init__(self):
        AFabricMachineLearning.__init__(self)
    
    def create(self, analysis_type, model_type, source_data, converted_data, air_pollution):
        if model_type == 'CatBoost':
            return CatBoostLearning(analysis_type, source_data, converted_data, air_pollution)
        
        elif model_type == 'XGBoost':
            return XGBoostLearning(analysis_type, source_data, converted_data, air_pollution)
        
        else:
            raise RuntimeError('Invalid model type passed to the task!')
