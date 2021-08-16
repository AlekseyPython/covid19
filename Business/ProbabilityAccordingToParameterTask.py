import pandas as pd
import numpy as np
from .ATask import ATask
from . import Fitter
import Initialization
from Entires.Enums import FittingCurve
from Entires.Selections import Selections


class Task(ATask):
    def __init__(self, signal_message, parameter, selections, fitting_curve):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.parameter = parameter
        self.selections = selections
        self.fitting_curve = fitting_curve
        self.optimal_parameters = None
    
    def run(self):
        selected_data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, self.parameter, self.selections)
        
        common_selection = Selections()
        for selection in self.selections:
            if selection.get_common():
                common_selection.add(selection)
            
        all_data = Initialization.icontroller_data_sourse.read_converted_data(self.signal_message, self.parameter, common_selection)
        
        name_parameter = self.parameter.get_name()
        unique_all_data = all_data[name_parameter].value_counts(sort=False)
        unique_selected_data = selected_data[name_parameter].value_counts(sort=False)
        
        df = pd.DataFrame({'all':unique_all_data, 'selected':unique_selected_data})
        df['probability'] = df['selected'] / df['all']
        df = df[df['probability'].notnull()]
        
        optimal_parameters, R2 = self._fill_optimal_parameters(df)
        
        df['probability'] *= 100
        df = df.drop(['all', 'selected'], axis=1)
        
        self.result = {}
        self.result['optimal_parameters'] = optimal_parameters
        self.result['R2'] = R2
        self.result['data'] = df
        self.result['size_selection'] = len(selected_data)
        
    def _fill_optimal_parameters(self, df):
        if not self.fitting_curve:
            return 
        
        sizes_selections = df['all'].to_numpy()
        x = np.array(df.index)
        y = df['probability'].to_numpy()
            
        if self.fitting_curve == FittingCurve.gompertz.value:
            fitter = Fitter.GompertzFitter(sizes_selections)   
            return fitter.fit(x, y)
        
        
        
    
    
        