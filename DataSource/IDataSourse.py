from ControllerDataSource.ADataSource import ADataSourse
from .ReaderCSV import ReaderCSV
from .HDF5 import HDF5
from .GeoData import GeoData


class IDataSourse(ADataSourse):
    def __init__(self):
        ADataSourse.__init__(self)
        
    def read_csv_data(self, file_name, signal_message, columns):
        reader = ReaderCSV(file_name, signal_message, columns)  
        return reader.read() 
         
    def read_hdf5_data(self, file_name, signal_message, table_name='ConvertedData', columns=None):
        reader = HDF5(file_name, signal_message)
        return reader.read(table_name, columns)
    
    def write_hdf5_data(self, file_name, signal_message, data, table_name='ConvertedData'):
        writer = HDF5(file_name, signal_message)
        return writer.write(data, table_name)
    
    def read_geo_data(self, signal_message, file_name):
        geo_data = GeoData(signal_message, file_name)
        return geo_data.read()
        
    def write_geo_data(self, signal_message, file_name, data):
        geo_data = GeoData(signal_message, file_name)
        return geo_data.write(data)
    
        
