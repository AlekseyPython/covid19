import sys
from PyQt5 import QtWidgets
from Presentation import MainWindowForm
from Presentation import ShowTextForm
from .Matplotlib import Plotter, RegionPlotter, GeoPlotter
from .MessageBox import MessageBox
from ControllerPresentation.APresentation import APresentation


class IPresentation(APresentation):
    def __init__(self):
        APresentation.__init__(self)
        self.app = QtWidgets.QApplication(sys.argv)
        
    def build_mian_window(self):
        main_window = MainWindowForm.Form()
        main_window.show()
        
        returned = self.app.exec_()
        sys.exit(returned)
    
    def message_to_user(self, type_message, text, informative_text=''):
        message_box = MessageBox()
        message_box.show_message(type_message, text, informative_text)
        
    def show_text_in_widget(self, text, title):
        window = ShowTextForm.Form(text, title)
        window.show_window()
        return window
    
    def build_bar(self, data, xlabel='', ylabel='', title='', size_selection=0, data_selections=None, fig=None):
        plotter = Plotter(data, fig)
        return plotter.build_bar(xlabel, ylabel, title, size_selection, data_selections)
        
    def plot(self, data, xlabel='', ylabel='', title='', size_selection=0, selections=None, fig=None):
        plotter = Plotter(data, fig)
        return plotter.plot(xlabel, ylabel, title, size_selection, selections)
    
    def plot_curve(self, fitting_curve, optimal_parameters, R2, fig=None):
        plotter = Plotter(data=None, fig=fig)
        return plotter.plot_aproximation_curve(fitting_curve, optimal_parameters, R2)
        
    def create_region_plotter(self, region, background_data=None):
        return RegionPlotter(region, background_data)
    
    def show_region(self, region_plotter):
        region_plotter.show()
        
    def slide_show_of_geo_data(self, data, column_slide_separator, title, region_plotter, slide_delay=1, selections=None, save_pictures=False, show_data_only_in_region=True):
        slide_plotter = GeoPlotter(data, title, region_plotter, selections, show_data_only_in_region)
        return slide_plotter.slide_show_scatter(column_slide_separator, slide_delay, save_pictures)
    
    def show_geo_data(self, data, title, region_plotter, selections=None, show_data_only_in_region=True):
        slide_plotter = GeoPlotter(data, title, region_plotter, selections, show_data_only_in_region)
        return slide_plotter.scatter()
    
    def plot_trisurf(self, data, box_aspect, title='', size_selection=0, data_selections=None, fig=None):
        plotter = Plotter(data, fig)
        return plotter.plot_trisurf(box_aspect, title, size_selection, data_selections)
    
    def scatter_3D(self, data, title='', size_selection=0, data_selections=None, fig=None):
        plotter = Plotter(data, fig)
        return plotter.scatter_3D(title, size_selection, data_selections)
    