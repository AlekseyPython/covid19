from Business.AControllerDataSourse import AControllerDataSourse
from Entires.Selections import Selections
from Entires import ConvertedColumns
import Initialization, Settings


class IControllerDataSourse(AControllerDataSourse):
    def __init__(self): pass
    
    def read_source_data(self, signal_message, columns=None):
        from ControllerDataSource.SourceDataController import Controller
        
        source_data = Controller(signal_message, columns)
        return source_data.get_source_data()
        
    def convert_data(self, signal_message):
        from ControllerDataSource.ConverterCsvToHdf5Controller import Controller
        
        source_data = self.read_source_data(signal_message)
        converter = Controller(signal_message, source_data)
        return converter.convert()
    
    def read_converted_data(self, signal_message, columns=None, selections=Selections(), ignore_empty_values=True):
        from ControllerDataSource.ConvertedDataController import Controller
        if isinstance(columns, ConvertedColumns._Column):
            columns = [columns]
            
        controller = Controller(signal_message, columns, selections, ignore_empty_values)
        return controller.get_data()
    
    def read_prepared_data(self, signal_message, table_name, columns=None):
        file_name = Settings.CONVERTED_DATA
        return Initialization.idata_source.read_hdf5_data(file_name, signal_message, table_name, columns)
    
    def write_prepared_data(self, signal_message, data, table_name):
        file_name = Settings.CONVERTED_DATA
        return Initialization.idata_source.write_hdf5_data(file_name, signal_message, data, table_name)
    
    def read_geo_data(self, signal_message, file_name):
        Initialization.idata_source.read_geo_data(signal_message, file_name)
        
    def write_geo_data(self, signal_message, file_name, data):
        Initialization.idata_source.write_geo_data(signal_message, file_name, data)
