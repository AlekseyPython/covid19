import geopandas as gpd
from Entires.FilesChecker import FilesChecker


class GeoData:
    def __init__(self, signal_message, file_name):
        self.signal_message = signal_message
        self.file_name = file_name
        
    def read(self):
        checker = FilesChecker(self.signal_message)
        if not checker.existence(self.file_name):
            return None
        
        return gpd.read_file(self.file_name)
        
    def write(self, data):
        data.to_file(self.file_name)
        return True
