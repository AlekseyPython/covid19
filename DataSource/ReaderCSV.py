import numpy as np
from Entires.SourceColumns import _Column as SourceColumn 
from Entires.FilesChecker import FilesChecker


class ReaderCSV:
    def __init__(self, file_name, signal_message, columns):
        self.file_name = file_name
        self.signal_message = signal_message
        self.columns = columns
    
    def read(self):
        checker = FilesChecker(self.signal_message)
        if not checker.existence(self.file_name):
            return False
        
        #pandas 1.3.1 skips 18 arbitrary lines when loading, so it's better not to use it. Use numpy!
        dtype = self._get_dtype()
        usecols = self._get_positions()
        data = np.loadtxt(self.file_name, dtype, comments=None, delimiter='#', skiprows=1, usecols=usecols)
        return data   
          
    def _get_dtype(self):
        names = map(SourceColumn.get_name, self.columns)
        types = map(SourceColumn.get_type, self.columns)
        dtypes = list(zip(names, types))
        return np.dtype(dtypes)
    
    def _get_positions(self):
        positions = list(map(SourceColumn.get_position, self.columns))
        return tuple(positions)        