from abc import ABCMeta, abstractmethod
from Entires.Selections import Selections


class AModel(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod
    def set_parameter(self, name, value): pass
    
    @abstractmethod
    def fit(self, data, silently=False): pass
    
