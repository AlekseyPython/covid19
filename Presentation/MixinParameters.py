from PyQt5.QtCore import QVariant
from Entires import SourceColumns as SC, ConvertedColumns as CC, SimilarColumns as SimC


class MixinParameters:
    def __init__(self):
        if not hasattr(self, 'controller'):
            return
        
        self._fill_combobox_parameters()
        if hasattr(self, 'register_procedure'):
            self.register_procedure(self._set_parameters)
    
    def _fill_combobox_parameters(self):
        parameters_with_values = self.controller.get_possible_values_parameters()
        for index, name in enumerate(parameters_with_values.keys()):
            label = getattr(self, 'LabelParameter' + str(index))
            label.setText(name + ':')
            
            combobox = getattr(self, 'ComboBoxParameter' + str(index))
            combobox.clear()
            
            possible_values = parameters_with_values[name]
            if type(possible_values[0]) in [SC._Column, CC._Column, SimC.SimilarColumns]:
                possible_values = sorted(possible_values, key=lambda x: x.get_name())
            # else:
            #     possible_values = sorted(possible_values)
                
            for column in possible_values:
                if type(column) == str:
                    name = column
                else:
                    name = column.get_name()
                combobox.addItem(name, QVariant(column))
            combobox.setCurrentIndex(0)
            
    def get_parameters(self):
        parameters = {}
        possible_values = self.controller.get_possible_values_parameters()
        for index, name in enumerate(possible_values.keys()):
            combobox = getattr(self, 'ComboBoxParameter' + str(index))
            parameters[name] = combobox.currentData()
        return parameters
    
    def _set_parameters(self):
        parameters = self.get_parameters()
        self.controller.set_parameters(parameters)
        
        