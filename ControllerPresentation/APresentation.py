from abc import ABCMeta, abstractmethod

class APresentation(metaclass=ABCMeta):
    def __init__(self): pass
    
    @abstractmethod
    def message_to_user(self, type_message, text, informative_text=''): pass
    
    @abstractmethod
    def show_text_in_widget(self, text, title): pass
    
    @abstractmethod
    def plot(self, data, xlabel='', ylabel='', title='', size_selection=0, data_selections=None): pass
     
    @abstractmethod   
    def create_region_plotter(self, region, background_data=None, background_size_point=1, background_color='gold'): pass
    
    @abstractmethod
    def show_region(self, region_plotter): pass
    
    @abstractmethod   
    def slide_show_of_geo_data(self, data, column_slide_separator, title, region_plotter, slide_delay=1, colour_column=None, selections=None, size_point=1, save_pictures=False): pass