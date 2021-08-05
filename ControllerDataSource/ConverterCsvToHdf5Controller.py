import multiprocessing
import pandas as pd
from pandas import CategoricalDtype
from . import Converters
from Entires import Functions
import Initialization, Settings


class Controller:
    def __init__(self, signal_message, source_data):
        self.signal_message = signal_message
        self.source_data = source_data
        self.file_hdf5 = Settings.CONVERTED_DATA
    
    def convert(self):
        converting_columns = Functions.get_columns(converted=True)
        data_pd = self._convert_numpy_array_to_dataframe(self.source_data, converting_columns)
        Initialization.idata_source.write_hdf5_data(self.file_hdf5, self.signal_message, data_pd, table_name='ConvertedData')
        return True
    
    def _convert_numpy_array_to_dataframe(self, data_np, pd_columns):
        all_converted_data = {}
        for column in pd_columns:
            converter = Converters.get_converter(column)
            
            with multiprocessing.Pool() as pool:
                values = pool.map(converter, data_np)
             
            dtype = self._get_dtype_column(column)
            converted_column = pd.Series(values, name=column.get_name(), dtype=dtype)
            all_converted_data[column.get_name()] = converted_column
            
        return pd.DataFrame(all_converted_data)
        
    @staticmethod
    def _get_dtype_column(column):
        dtype = column.get_type()
        if dtype == 'category':
            categories, odered = column.get_categories()
            return CategoricalDtype(categories=categories, ordered=odered)
        
        elif dtype in ['bool', 'int8']:
            return 'int8'
        
        else:
            return dtype
    
    
    
    