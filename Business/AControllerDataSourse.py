from abc import ABCMeta, abstractmethod
from Entires.Selections import Selections


class AControllerDataSourse(metaclass=ABCMeta):
    def __init__(self):pass
    
    @abstractmethod    
    def read_source_data(self, signal_message, columns):pass
     
    @abstractmethod     
    def convert_data(self, signal_message):pass
    
    @abstractmethod
    def read_converted_data(self, signal_message, columns, selections=Selections(), ignore_empty_values=True):pass
    
    @abstractmethod
    def read_prepared_data(self, signal_message, table_name, columns=None): pass
    
    @abstractmethod
    def write_prepared_data(self, signal_message, data, table_name): pass
    
    @abstractmethod
    def read_geo_data(self, signal_message, file_name): pass
    
    @abstractmethod    
    def write_geo_data(self, signal_message, file_name, data): pass