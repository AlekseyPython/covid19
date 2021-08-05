# import cv2
import os
import copy
import time
from platform import system
import pandas as pd
import numpy as np
import imageio
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from pandas.core.indexes.datetimes import DatetimeIndex
from Entires.Enums import FittingCurve
import Settings


class Plotter:
    def __init__(self, data=None, fig=None):
        self.data = data
        self.fig = fig
        sns.set_style('darkgrid')
    
    def get_ax(self):
        if self.fig is None:
            self.fig = plt.figure()
            
        axes = self.fig.axes
        if len(axes) > 0:
            return axes[len(axes)-1]
        return None
            
    def plot_trisurf(self, box_aspect, title, size_selection, data_selections):
        ax = self.get_ax()
        if ax is None:
            ax = Axes3D(self.fig)
            
        ax.set_box_aspect(box_aspect)
        self._plot_titles(ax, title, size_selection, data_selections, combine=True)
        
        ax.plot_trisurf(self.data.X, self.data.Y, self.data.Z, cmap=cm.coolwarm, linewidth=0.2, alpha=0.4, zorder=0)
        plt.show()
        return self.fig
    
    def scatter_3D(self, title, size_selection, data_selections):
        ax = self.get_ax()
        if ax is None:
            ax = Axes3D(self.fig)
        
        self._plot_titles(ax, title, size_selection, data_selections, combine=True)    
        ax.scatter(self.data.X, self.data.Y, self.data.Z, s=50, c='black', alpha=1, zorder=3)
        plt.show()
        return self.fig
                
    def build_bar(self, *args, **kwargs):
        return self._plot_data(kind='bar', *args, **kwargs)
            
    def plot(self, *args, **kwargs):
        return self._plot_data(kind='line', *args, **kwargs)
    
    def plot_curve(self, fitting_curve, optimal_parameters, R2):
        x = np.array(self.data.index)
        if fitting_curve == FittingCurve.gompertz.value:
            a, b, c = optimal_parameters
            y = a*np.exp(b*x) + c
            y *= 100
        else:
            raise RuntimeError('For building the approximating curve, an unknown function is passed!')
        
        #write equation
        x_text = x[0]
        y_text = y[-1]/2
        equation_text = 'y = ' + str(c) +' + ' + str(round(a, 6)) + ' * exp(' + str(round(b, 6)) + ' * Age)\n'   
        equation_text += 'R2 = ' + str(round(R2, 4))
        
        ax = self.get_ax()
        ax.text(x_text, y_text, equation_text, fontsize=10)
        ax.plot(x, y)
        self._set_locator_for_xasis(ax)
              
    def _plot_data(self, kind, xlabel, ylabel, title, size_selection, data_selections):
        ax = self.get_ax()
        if ax is None:
            ax = self.fig.add_axes()
            
        self.fig.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.91)
        self._plot_titles(ax, title, size_selection, data_selections)
        
        if len(self.data) == 0:
            plt.show()
            return self.fig
        
        self._set_locator_for_xasis(ax)
        self._cut_out_empty_time_from_dates()
        rotation = self._get_rotation_set_xticklabels()
        self.data.plot(kind=kind, ax=ax, xlabel=xlabel, ylabel=ylabel, rot=rotation)
            
        #it is necessary to call a second time, so that the labels are correctly reflected for all charts     
        self._set_locator_for_xasis(ax)
        
        plt.show()    
        return self.fig
    
    def _plot_titles(self, ax, title, size_selection, data_selections, combine=False):
        if not title:
            return 
        
        title = self._get_title(title, size_selection)
        second_title = self._get_sub_title(data_selections)
        
        if combine and second_title:
            self.fig.suptitle(title + '\n' + second_title, fontsize=12)
            return
                
        self.fig.suptitle(title, fontsize=12)
        if second_title:
            ax.set_title(second_title, fontsize=10)
            
    def _set_locator_for_xasis(self, ax):
        def get_step_ticker():
            step_ticker = round(len(self.data) / 40)
            return max(1, step_ticker)
    
        step_ticker = get_step_ticker()
        locator = ticker.MultipleLocator(step_ticker)
        ax.xaxis.set_major_locator(locator)
        
    def _cut_out_empty_time_from_dates(self):  
        if type(self.data.index) != DatetimeIndex:
            return
        
        self.data.sort_index(inplace=True)
        new_index = list(map(lambda x: str(x)[:10], self.data.index))
        self.data.index = new_index
             
    def _get_title(self, title, size_selections):
        title += ' (n= ' + str(size_selections)
        
        if len(self.data)>0 and type(self.data) != pd.DataFrame:
            if self.data.index.dtype.name.find('int') >= 0:
                mean = 0
                for value, probability in self.data.items():
                    mean += probability * value / 100
                    
                dispersion  = 0
                for value, probability in self.data.items():
                    dispersion += probability * pow(value - mean, 2) / 100
                std = np.sqrt(dispersion)
            else:
                mean = self.data.mean()
                #std = self.data.std() wrong value!!!
                
                dispersion  = 0
                probability = 1 / len(self.data)
                for value in self.data:
                    dispersion += probability * pow(value - mean, 2)
                std = np.sqrt(dispersion)
            
            mean = round(mean, 2)
            std = round(std, 2)    
            title += ', mean=' + str(mean) + ', std= '+ str(std)
            
        title += ')'
        return title
    
    @staticmethod
    def _get_sub_title(data_selections):
        sub_title = str(data_selections)
        return sub_title.replace('T00:00:00.000000000', '')
    
    def _get_rotation_set_xticklabels(self):
        if self.data.index.dtype.name.find('int') >= 0:
            return 0
        elif len(self.data) < 10:
            return 15
        elif len(self.data) == 12:
            return 55
        else:
            return 90
        

class RegionPlotter:
    def __init__(self, region, background_data=None):
        self.region = region
        self.background_data = background_data
        
        sns.set_style(style='white')
        self.fig, self.ax = plt.subplots(1, 1)
       
    def show(self):
        self._build_boundaries()
        self._decorate_figure()
        self._remove_artifacts()
        self._build_background()
        plt.show()
    
    def get_region(self):
        return self.region
        
    def get_fig_and_ax(self):
        return self.fig, self.ax
        
    def _build_boundaries(self):
        boundaries_moscow = self.region.get_boundaries()
        boundaries_moscow.boundary.plot(ax=self.ax, zorder=-5)
        
    def _build_background(self):
        if self.background_data is None:
            return
        
        pointsInPolygon = self.region.get_points_in_region(self.background_data)
        
        parameters = {}
        if 'marker' not in pointsInPolygon:
            pointsInPolygon['marker'] = 'o'
            
        markers = pointsInPolygon['marker'].unique()    
        for marker in markers:
            data_marker = pointsInPolygon[pointsInPolygon['marker']==marker]
            
            parameters['x'] = data_marker['Longitude']
            parameters['y'] = data_marker['Latitude']
            parameters['marker'] = marker
            
            if 'color' in pointsInPolygon:
                parameters['c'] = data_marker['color']
            else:
                parameters['c'] = 'gold'
                
            if 'shape' in pointsInPolygon:
                parameters['s'] = data_marker['shape']
            else:
                parameters['s'] = 1
            
            self.ax.scatter(**parameters)
    
    def _decorate_figure(self):
        self.fig.subplots_adjust(left=0.05, bottom=0.03, right=0.97, top=0.95)
        # self.ax.xaxis.set_major_formatter(plt.NullFormatter())
        # self.ax.yaxis.set_major_formatter(plt.NullFormatter())
        
        aspect = self.region.get_aspect()
        self.ax.set_aspect(aspect)
        # plt.axis('off')
        
        xlim = self.region.get_xlim()
        if xlim is not None:
            self.ax.set_xlim(xlim)
        
        ylim = self.region.get_ylim()
        if ylim is not None:     
            self.ax.set_ylim(ylim) 
        
    def _remove_artifacts(self):
        self.ax.add_patch(plt.Circle((37.707, 55.654), 0.02, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.733, 55.685), 0.002, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.695, 55.7125), 0.015, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.475, 55.7150), 0.025, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.51, 55.76), 0.01, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.587, 55.905), 0.003, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.55, 55.865), 0.025, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.65, 55.855), 0.025, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.83, 55.657), 0.007, color='white', zorder=-3))
        self.ax.add_patch(plt.Circle((37.376, 55.911), 0.004, color='white', zorder=-3))
        self.ax.add_patch(plt.Rectangle((37.78154, 55.80937), 0.05667, 0.00481, color='white', zorder=-3))


class GeoPlotter:
    def __init__(self, data, title, region_plotter, selections=None, show_data_only_in_region=True):
        self.data = data
        self.title = title
        self.region_plotter = region_plotter
        self.selections = selections
        self.show_data_only_in_region = show_data_only_in_region
    
    def slide_show_scatter(self, column_slide_separator, slide_delay=1, save_pictures=False):
        self._maximize_window()
        self._build_points(column_slide_separator, slide_delay, save_pictures)
        
        if save_pictures:
            self._create_common_picture()
            self._create_common_video()
    
    def scatter(self):
        fig, ax = self.region_plotter.get_fig_and_ax()
        
        if self.show_data_only_in_region:
            region = self.region_plotter.get_region()
            data_in_region = region.get_points_in_region(self.data)
        else:
            data_in_region = self.data
            
        second_title = Plotter._get_sub_title(self.selections)
        title = self.title + '   ' + second_title
        fig.suptitle(title, fontsize=12)
        
        self._build_scatter_of_data(data_in_region, ax)
        plt.show()
    
    @staticmethod
    def _build_scatter_of_data(data, ax):  
        parameters = {}
        if 'marker' not in data:
            data['marker'] = 'o'
            
        markers = data['marker'].unique()    
        for marker in markers:
            data_marker = data[data['marker']==marker]
            
            parameters['x'] = data_marker['Longitude']
            parameters['y'] = data_marker['Latitude']
            parameters['marker'] = marker
            
            if 'color' in data_marker:
                parameters['c'] = data_marker['color']
            else:
                parameters['c'] = 'green'
                
            if 'shape' in data_marker:
                parameters['s'] = data_marker['shape']
            else:
                parameters['s'] = 1
                
            
            ax.scatter(**parameters)  
        
    @staticmethod 
    def _maximize_window():
        backend = plt.get_backend()
        cfm = plt.get_current_fig_manager()
        
        if backend == "wxAgg":
            cfm.frame.Maximize(True)
            
        elif backend == "TkAgg":
            if system() == "Windows":
                cfm.window.state("zoomed")  # This is windows only
            else:
                cfm.resize(*cfm.window.maxsize())
                
        elif backend in ["QT4Agg", "QT5Ag"]:
            cfm.window.showMaximized()
            
        elif callable(getattr(cfm, "full_screen_toggle", None)):
            if not getattr(cfm, "flag_is_max", None):
                cfm.full_screen_toggle()
                cfm.flag_is_max = True
                
        else:
            raise RuntimeError("plt_maximize() is not implemented for current backend:", backend) 
        
    def _build_points(self, column_slide_separator, slide_delay, save_pictures):
        fig, ax = self.region_plotter.get_fig_and_ax()
        
        if self.show_data_only_in_region:
            region = self.region_plotter.get_region()
            data_in_region = region.get_points_in_region(self.data)
        else:
            data_in_region = self.data
            
        slide_values = data_in_region[column_slide_separator].unique()
        slide_values = sorted(slide_values)
        
        accumulated_data = 0    
        for slide_value in slide_values:
            slide_data = data_in_region[data_in_region[column_slide_separator]==slide_value]
            accumulated_data += len(slide_data)
            
            selections = copy.deepcopy(self.selections)
            selections.add_value(column_name=column_slide_separator, value=slide_value)
            
            second_title = Plotter._get_sub_title(selections)
            second_title = second_title.replace('DateCreating', 'Date')
            title = self.title + '   ' + second_title + ' (n= ' + str(len(slide_data)) + ' / ' + str(accumulated_data) + ')'
            fig.suptitle(title, fontsize=12)
            
            self._build_scatter_of_data(slide_data, ax)
            plt.show()
            
            if save_pictures:
                plt.savefig(Settings.FOLDER_WITH_IMAGES + '/' + second_title + '.jpeg', optimize=True, progressive=True, quality=90)
                
            time.sleep(slide_delay)
            
    @staticmethod
    def _create_common_picture():
        file_names = []
        for (_, _, file_names) in os.walk(Settings.FOLDER_WITH_IMAGES):
            file_names.extend(file_names)
            break
        
        images = []
        for file_name in file_names:
            images.append(imageio.imread(Settings.FOLDER_WITH_IMAGES + '/' + file_name))
        imageio.mimsave(Settings.FOLDER_WITH_IMAGES + '/Moscow.gif', images, duration=3)
    
    @staticmethod
    def _create_common_video():pass
        # image_folder = Settings.FOLDER_WITH_IMAGES
        # video_name = 'video.avi'
        #
        # images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        # height, width, _ = frame.shape
        #
        # video = cv2.VideoWriter(Settings.FOLDER_WITH_IMAGES + '/' + video_name, 0, 1, (width,height))
        #
        # for image in images:
        #     video.write(cv2.imread(os.path.join(image_folder, image)))
        #
        # cv2.destroyAllWindows()
        # video.release()
    
    