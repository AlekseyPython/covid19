import pandas as pd
from Entires.ConvertedColumns import _Column as ConvertedColumn
from Entires.FilesChecker import FilesChecker


class HDF5:
    def __init__(self, file_name, signal_message):
        self.file_name = file_name
        self.signal_message = signal_message
    
    def write(self, df, table_name='ConvertedData'):
        df.to_hdf(self.file_name, key=table_name, mode='a', complevel=4, append=False, format='table')
        return True
    
    def read(self, table_name='ConvertedData', columns=None):
        preparer = FilesChecker(self.signal_message)
        if not preparer.existence(self.file_name):
            return None
        
        if columns is not None and type(columns[0]) == ConvertedColumn:
            columns = list(map(ConvertedColumn.get_name, columns))
            
        return pd.read_hdf(self.file_name, key=table_name, mode='r', columns=columns)
        
    
