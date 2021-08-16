from enum import Enum
from collections import namedtuple

   
class FittingCurve(Enum):
    gompertz = 'Gompertz distribution'
    
    
class MoscowAreas(Enum):
    old_moscow  = 'Old part of Moscow'
    new_mMoscow =  'New part of Moscow'
    zelenograd  = 'Zelenograd'
    all_moscow  = 'All Moscow'
    
    
class PeriodsPollution(Enum):
    last_month  = 'Last month'
    last_year   =  'Last year'
    
    
class TablesHDF5(Enum):
    ConvertedData = 'ConvertedData'
    AirPolutionsForLastYear = 'AirPolutionsForLastYear'
    AirPolutionsForLastMonth= 'AirPolutionsForLastMonth'


class AnalysisTypes(Enum):  
    PredictionSeverityOfDisease                     = 'Prediction severity of disease'
    ObtainingCharacteristicsOfSeriouslyDiseasePeople= 'Obtaining characteristics of seriously disease people'
    
    
class OperationsOfMachineLearning(Enum):
    NeuralNetworkTraining   = 'Neural network training on text data '
    FeaturesProfiling       = 'Features profiling'
    AvoidanceOverfitting    = 'Avoidance overfitting'
    SearchEliminatedFeatures= 'Search eliminated_features'
    SearchPenalties         = 'Search penalties'
    SearchHyperparameters   = 'Search hyperparameters'
    OptimizeTrainTestSplit  = 'Optimize ratio train and test split'
    OptimizeLearningRate    = 'Optimize learning rate'
    AllOperationsInCircle   = 'All operations in a circle'
    
#need_to_continue_optimization does't make sense for all operations     
ResultOperationOfMachineLearning = namedtuple('ResultOperationOfMachineLearning' , ['operation', 'parameters', 'metrics', 'need_to_continue_optimization'], defaults=(None,))
ParametersTemporaryFiles = namedtuple('ParametersTemporaryFiles' , ['analysis_type', 'operation', 'type_machine_learning_model', 'severity_of_disease'])
    

class TypesOfGradientBoostingModel(Enum):
    CatBoost = 'CatBoost'
    XGBoost  = 'XGBoost (not implemented)'


def get_values_enum(cls_enum):
    return [e.value for e in cls_enum]
      
def get_table_hdf5_by_period_pollution(period_pollution):
    if period_pollution in [PeriodsPollution.last_month, PeriodsPollution.last_month.value]:
        return TablesHDF5.AirPolutionsForLastMonth.value
    
    elif period_pollution in [PeriodsPollution.last_year, PeriodsPollution.last_year.value]:
        return TablesHDF5.AirPolutionsForLastYear.value
    
    else:
        raise RuntimeError('Please, use the pollution calculation period for which there is exist table in the file HDF5')