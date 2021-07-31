from abc import ABCMeta, abstractmethod


class AMachineLearning(metaclass=ABCMeta):
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
    
    
class AFabricMachineLearning(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod
    def create(self):pass
        
