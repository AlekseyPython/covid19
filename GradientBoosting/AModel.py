from abc import ABCMeta, abstractmethod


class AModel(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod
    def set_parameter(self, name, value): pass
    
    @abstractmethod
    def fit(self, data, silently=False): pass
    
