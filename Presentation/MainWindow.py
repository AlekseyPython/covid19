from PyQt5 import QtWidgets, uic


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self, parent)
    
        uic.loadUi('Presentation/Forms/MainWindow.ui', self)
         
        #name of program
        self.setWindowTitle('Covid- 19 reseaches by moscow database')
          
        #Menu
        menu = self.menuBar()
        sub_menu = menu.addMenu('&Data preparation')
        self._add_menu_item(sub_menu, 'Create Moscow regions maps', self._show_dialog_create_moscow_regions_maps)
        self._add_menu_item(sub_menu, 'Get general list values', self._show_dialog_get_general_list_values)
        self._add_menu_item(sub_menu, 'Convert data to desired types', self._show_dialog_convert_data_to_desired_types)
        self._add_menu_item(sub_menu, 'Count air pollutions for Moscow citizens', self._show_dialog_count_air_pollutions_for_moscow_citizens)
        self._add_menu_item(sub_menu, 'Tuning a prediction model for disease severity', self._show_dialog_tuning_prediction_model)
        
        sub_menu = menu.addMenu('&Analysis')
        self._add_menu_item(sub_menu, 'Probability distribution', self._show_dialog_probability_distribution)
        self._add_menu_item(sub_menu, 'Time dependence', self._show_dialog_time_dependence)
        self._add_menu_item(sub_menu, 'Probability according to parameter', self._show_dialog_probability_according_to_parameter)
        self._add_menu_item(sub_menu, 'The spread of disease in Moscow', self._show_dialog_spread_of_disease_in_Moscow)
        self._add_menu_item(sub_menu, 'Air pollution in Moscow', self._show_dialog_air_pollution_in_moscow)
        self._add_menu_item(sub_menu, 'Finding robust parameters of the Gompertz law', self._show_dialog_finding_robust_parameters_of_gompertz_law)
        
    def _add_menu_item(self, parent_menu, name, func):
        action = QtWidgets.QAction('&' + name, self)
        action.triggered.connect(func)
        parent_menu.addAction(action)
    
    def _show_dialog_create_moscow_regions_maps(self):
        from ControllerPresentation.MoscowRegionsMaps import Controller
        from .MoscowRegionsMaps import MoscowRegionsMaps
        self._show_form(Controller, MoscowRegionsMaps)
        
    def _show_dialog_get_general_list_values(self):
        from ControllerPresentation.GeneralListValues import Controller
        from .GeneralListValues import GeneralListValues
        self._show_form(Controller, GeneralListValues)
    
    def _show_dialog_convert_data_to_desired_types(self):
        from ControllerPresentation.ConvertDataToDesiredTypes import Controller
        from .ConvertDataToDesiredTypes import ConvertDataToDesiredTypes
        self._show_form(Controller, ConvertDataToDesiredTypes)
    
    def _show_dialog_count_air_pollutions_for_moscow_citizens(self):
        from ControllerPresentation.CountAirPollutionsForMoscowCitizens import Controller
        from .CountAirPollutionsForMoscowCitizens import CountAirPollutionsForMoscowCitizens
        self._show_form(Controller, CountAirPollutionsForMoscowCitizens)
            
    def _show_dialog_tuning_prediction_model(self):
        from ControllerPresentation.TuningPredictionModel import Controller
        from .TuningPredictionModel import TuningPredictionModel
        self._show_form(Controller, TuningPredictionModel)
    
    def _show_dialog_probability_distribution(self):
        from ControllerPresentation.ProbabilityDistribution import Controller
        from .ProbabilityDistribution import ProbabilityDistribution
        self._show_form(Controller, ProbabilityDistribution)
            
    def _show_dialog_time_dependence(self):
        from ControllerPresentation.TimeDependence import Controller
        from Presentation.TimeDependence import TimeDependence
        self._show_form(Controller, TimeDependence)
    
    def _show_dialog_probability_according_to_parameter(self):
        from ControllerPresentation.ProbabilityAccordingToParameter import Controller
        from .ProbabilityAccordingToParameter import ProbabilityAccordingToParameter
        self._show_form(Controller, ProbabilityAccordingToParameter)
        
    def _show_dialog_spread_of_disease_in_Moscow(self):
        from ControllerPresentation.SpreadOfDiseaseInMoscow import Controller
        from .SpreadOfDiseaseInMoscow import SpreadOfDiseaseInMoscow
        self._show_form(Controller, SpreadOfDiseaseInMoscow)
        
    def _show_dialog_air_pollution_in_moscow(self):
        from ControllerPresentation.AirPollutionInMoscow import Controller
        from .AirPollutionInMoscow import AirPollutionInMoscow
        self._show_form(Controller, AirPollutionInMoscow)
    
    def _show_dialog_finding_robust_parameters_of_gompertz_law(self):pass
        # from ControllerPresentation.FindingRobustParametersOfGompertzLaw import Controller
        # from Presentation import AButton
        # self._show_form(Controller, AButton)
        
    def _show_form(self, Controller, Form):
        controller = Controller()
        form = Form(controller)
        form.show()
        
        if not hasattr(self, 'forms'):
            self.forms = []
            
        self.forms.append(form)
            
        
            
        
        
        
    
