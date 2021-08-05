import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import Settings


class GeoRegion:
    def __init__(self, name, names_shape_files):
        self.name = name
        self.boundaries = self._count_boundaries(names_shape_files)
    
    def get_name(self):
        return self.name
    
    def get_xlim(self):
        if self.name == 'Old part of Moscow':
            return 37.18, 37.99
        else:
            return None
        
    def get_ylim(self):
        if self.name == 'Old part of Moscow':
            return 55.48, 55.96
        else:
            return None
    
    def get_aspect(self):
        if self.name == 'Old part of Moscow':
            return 1.7
        else:
            return 1
         
    def get_points_in_region(self, pd_points):  
        pd_points['geometry'] = list(map(Point, pd_points['Longitude'], pd_points['Latitude']))
        geo_points = gpd.GeoDataFrame(pd_points)
        return gpd.sjoin(geo_points, self.boundaries)
    
    def get_boundaries(self):
        return self.boundaries
        
    def _count_boundaries(self, names_shape_files):
        boundaries=[]
        for name_file in names_shape_files:
            shape_file = Settings.FOLDER_MOSCOW_MAPS + '/' + name_file + '.shp'
            separated_by_regions = gpd.read_file(shape_file)
           
            poligons = []
            for i in range(len(separated_by_regions)):
                poligon = separated_by_regions.iloc[i]['geometry']
                poligons.append(poligon)
            
            combine_regions = unary_union(poligons)
            if type(combine_regions) == Polygon:
                combine_regions = MultiPolygon([combine_regions])
                 
            common_region = gpd.GeoDataFrame({'geometry':combine_regions})
            boundaries.extend(common_region['geometry'].to_numpy())
            
        return gpd.GeoDataFrame({'geometry':boundaries})
    
    
    
