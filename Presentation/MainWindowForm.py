from PyQt5 import QtWidgets, uic


class Form(QtWidgets.QMainWindow):
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
        from ControllerPresentation.MoscowRegionsMapsController import Controller
        from Presentation.MoscowRegionsMapsForm import Form
        self._show_form(Controller, Form)
        
    def _show_dialog_get_general_list_values(self):
        from ControllerPresentation.GeneralListValuesController import Controller
        from Presentation.GeneralListValuesForm import Form
        self._show_form(Controller, Form)
    
    def _show_dialog_convert_data_to_desired_types(self):
        from ControllerPresentation.ConvertDataToDesiredTypesController import Controller
        from Presentation.ConvertDataToDesiredTypesForm import Form
        self._show_form(Controller, Form)
    
    def _show_dialog_count_air_pollutions_for_moscow_citizens(self):
        from ControllerPresentation.CountAirPollutionsForMoscowCitizensController import Controller
        from Presentation.CountAirPollutionsForMoscowCitizensForm import Form
        self._show_form(Controller, Form)
            
    def _show_dialog_tuning_prediction_model(self):
        from ControllerPresentation.TuningPredictionModelController import Controller
        from Presentation.TuningPredictionModelForm import Form
        self._show_form(Controller, Form)
    
    def _show_dialog_probability_distribution(self):
        from ControllerPresentation.ProbabilityDistributionController import Controller
        from Presentation.ProbabilityDistributionForm import Form
        self._show_form(Controller, Form)
            
    def _show_dialog_time_dependence(self):
        from ControllerPresentation.TimeDependenceController import Controller
        from Presentation.TimeDependenceForm import Form
        self._show_form(Controller, Form)
    
    def _show_dialog_probability_according_to_parameter(self):
        from ControllerPresentation.ProbabilityAccordingToParameterController import Controller
        from Presentation.ProbabilityAccordingToParameterForm import Form
        self._show_form(Controller, Form)
        
    def _show_dialog_spread_of_disease_in_Moscow(self):
        from ControllerPresentation.SpreadOfDiseaseInMoscowController import Controller
        from Presentation.SpreadOfDiseaseInMoscowForm import Form
        self._show_form(Controller, Form)
        
    def _show_dialog_air_pollution_in_moscow(self):
        from ControllerPresentation.AirPollutionInMoscowController import Controller
        from Presentation.AirPollutionInMoscowForm import Form
        self._show_form(Controller, Form)
    
    def _show_dialog_finding_robust_parameters_of_gompertz_law(self):pass
        # from ControllerPresentation.FindingRobustParametersOfGompertzLawController import Controller
        # from Presentation import AButton
        # self._show_form(Controller, AButton)
        
    def _show_form(self, Controller, Form):
        controller = Controller()
        form = Form(controller)
        form.show()
        
        if not hasattr(self, 'forms'):
            self.forms = []
            
        self.forms.append(form)
            
        
            
        
        
        
    
