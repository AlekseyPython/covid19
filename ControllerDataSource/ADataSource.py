from abc import ABCMeta, abstractmethod


class ADataSourse(metaclass=ABCMeta):
    def __init__(self): pass
    
    @abstractmethod    
    def read_csv_data(self, file_name, signal_message, columns): pass
    
    @abstractmethod
    def read_hdf5_data(self, file_name, signal_message, columns, table_name='ConvertedData'): pass
    
    @abstractmethod
    def write_hdf5_data(self, file_name, signal_message, data, table_name='ConvertedData'): pass

    @abstractmethod
    def read_geo_data(self, signal_message, file_name): pass
     
    @abstractmethod    
    def write_geo_data(self, signal_message, file_name, data): pass