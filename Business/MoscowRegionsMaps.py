from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from Entires.Enums import MoscowAreas
from .ATask import ATask
import Initialization, Settings


class Task(ATask):
    def __init__(self, signal_message):
        ATask.__init__(self)
        self.signal_message = signal_message
    
    def run(self):
        self.result = {}
        self.result['data'] = self._create()
        
    def _create(self):
        all_moscow = self._prepare_all_moscow_geo()
        if all_moscow is None:
            return False
        
        for area_enum in MoscowAreas:
            area = area_enum.value
            if area == 'All Moscow':
                continue
        
            removing_features = []
            for index_row in all_moscow.index:
                row = all_moscow.loc[index_row]
                
                if area == 'Zelenograd':
                    if row['ABBREV_AO'] != 'ЗелАО':
                        removing_features.append(index_row)
                    
                elif area == 'New part of Moscow':
                    if row['ABBREV_AO'] not in ['Новомосковский', 'Троицкий']:
                        removing_features.append(index_row)
                        
                elif area == 'Old part of Moscow':
                    if row['ABBREV_AO'] in ['Новомосковский', 'ЗелАО', 'Троицкий']:
                        removing_features.append(index_row)
                    
            current_region = all_moscow.copy()
            current_region = current_region.drop(removing_features)
            
            new_path_to_file = Settings.FOLDER_MOSCOW_MAPS + '/' + area + '.shp'
            result = Initialization.icontroller_data_sourse.write_geo_data(self.signal_message, new_path_to_file, current_region)
            if result is None:
                return None
        return True
    
    def _prepare_all_moscow_geo(self):
        path_to_file = Settings.FOLDER_MOSCOW_MAPS + '/' + 'All Moscow.shp'
        all_moscow = Initialization.icontroller_data_sourse.read_geo_data(self.signal_message, path_to_file)
        if all_moscow is None:
            return None
        
        removing_rows = []
        for index_row in range(len(all_moscow)):
            row = all_moscow.iloc[index_row]
            
            min_lat, _, max_lat, max_long = row['geometry'].bounds
            if max_long>55.7 and min_lat<37:
                new_polygons = []
                all_polygons = list(row['geometry'])
                for polygon in all_polygons:
                    _, _, max_lat_p, _ = polygon.bounds
                    if max_lat_p>37.13:
                        new_polygons.append(polygon)
                    
                combine_regions = unary_union(new_polygons)
                if type(combine_regions) == Polygon:
                    combine_regions = MultiPolygon([combine_regions])
                    
                all_moscow['geometry'].iloc[index_row] = combine_regions
                
            elif max_lat>37.6 and max_long>56:
                removing_rows.append(index_row)
        
        all_moscow = all_moscow.drop(removing_rows)        
        return all_moscow
    