import os
import numpy as np
from numpy.lib import recfunctions
from Entires import SourceColumns, Functions
import Initialization, Settings


class Controller:
    def __init__(self, signal_message, columns=None, use_npz_data=True):
        if isinstance(columns, SourceColumns._Column):
            columns = [columns]
        elif columns is None:
            columns = Functions.get_columns(converted=False)
                
        if not use_npz_data:
            if SourceColumns.Repeat not in columns:
                self.add_column_repeat = True
                columns.append(SourceColumns.Repeat)
                
            if SourceColumns.DateCreating not in columns:
                self.add_column_date_creating = True
                columns.append(SourceColumns.DateCreating)
            
        self.signal_message = signal_message
        self.columns        = columns
        self.use_npz_data   = use_npz_data
    
    def save_source_data_to_npz(self, source_data):
        file_name_npz = Settings.PATH_COVID_PATIENTS_NPZ
        np.savez_compressed(file_name_npz, source_data=source_data)
        
    def get_source_data(self):
        file_name_npz = Settings.PATH_COVID_PATIENTS_NPZ
        if self.use_npz_data and os.path.exists(file_name_npz):
            loaded = np.load(file_name_npz)
            source_data = loaded['source_data']
            return self._drop_unnecessary_columns(source_data)
        
        file_name = Settings.PATH_COVID_PATIENTS 
        source_data = Initialization.idata_source.read_csv_data(file_name, self.signal_message, self.columns)
        source_data = self._remove_repeate_strings(source_data)
        source_data = self._delete_last_date(source_data)
        return source_data
    
    def _drop_unnecessary_columns(self, source_data):
        if not self.use_npz_data:
            raise RuntimeError('The use of this function is possible only when receiving data in npz format!')
        
        all_columns = recfunctions.get_names_flat(source_data.dtype)
        nedeed_columns = [column.get_name() for column in self.columns]
        unnecessary_columns = [column for column in all_columns if column not in nedeed_columns]
        source_data = recfunctions.drop_fields(source_data, unnecessary_columns)
        return source_data
            
    def _remove_repeate_strings(self, source_data):
        deleting_indexes = []
        for index in range(len(source_data)):
            if source_data[index]['Repeat'].strip():
                deleting_indexes.append(index)
        
        source_data = np.delete(source_data, deleting_indexes, 0)
        if hasattr(self, 'add_column_repeat') and self.add_column_repeat:
            source_data = recfunctions.drop_fields(source_data, 'Repeat')
        return source_data
    
    def _delete_last_date(self, source_data):
        last_index = len(source_data) - 1
        last_date = source_data[last_index]['DateCreating']
        source_data = source_data[source_data['DateCreating'] != last_date]
        
        if hasattr(self, 'add_column_date_creating') and self.add_column_date_creating:
            source_data = recfunctions.drop_fields(source_data, 'DateCreating')
        return source_data
            