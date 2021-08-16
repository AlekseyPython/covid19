from abc import ABCMeta, abstractmethod


class AOptimizatorGradientBoosting(metaclass=ABCMeta):
    def __init__(self): pass
        
    @abstractmethod 
    def avoidance_overfitting(self, severities_of_disease, silently=True): pass
    
    @abstractmethod
    def search_eliminated_features(self, severities_of_disease, silently=True): pass
    
    @abstractmethod    
    def search_penalties(self, severities_of_disease, silently=True): pass
    
    @abstractmethod
    def search_hyperparameters(self, severities_of_disease, silently=True): pass
    
    @abstractmethod
    def optimize_train_test_split(self, severity_of_disease, silently=True): pass
    
    @abstractmethod
    def optimize_learning_rate(self, severity_of_disease, silently=True): pass
    
    
class AOptimizatorNeuralNetwork(metaclass=ABCMeta):
    def __init__(self): pass
        
    @abstractmethod 
    def learn(self, severities_of_disease, silently=True): pass
    
        
class AFabricGradientBoosting(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod
    def create_optimizator(self, analysis_type, model_type, source_data, converted_data, air_pollution, 
                            calculation_expediency_threshold, limit_unresulting_calculations):pass
        
    
class AFabricNeuralNetwork(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod
    def create_optimizator(self, source_data, converted_data): pass