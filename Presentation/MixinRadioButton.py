class MixinRadioButton:
    def __init__(self):
        if not hasattr(self, 'controller'):
            return
        
        self._init_radio_button()
        if hasattr(self, 'register_procedure'):
            self.register_procedure(self._set_radio_button)
            
    def _init_radio_button(self):
        radio_buttons = self.controller.get_radio_buttons()
        for index, name_radio_button in enumerate(radio_buttons):
            radio_button = getattr(self, 'RadioButtonParameter' + str(index))
            radio_button.setText(name_radio_button)
            
            if index==0:
                radio_button.toggled["bool"].connect(self._radio_button_toggled)
                radio_button.setChecked(True)
            else:
                radio_button.setChecked(False)
                
    def _get_radio_button_values(self):
        values = {}
        radio_buttons = self.controller.get_radio_buttons()
        for index, name_radio_button in enumerate(radio_buttons):
            radio_button = getattr(self, 'RadioButtonParameter' + str(index))
            values[name_radio_button] = radio_button.isChecked()
        return values
            
    def _radio_button_toggled(self, status):
        values = self._get_radio_button_values()
        self.controller.set_radio_button(values)
        
        if hasattr(self, 'ComboBoxParameter0'):
            self._fill_combobox_parameters()
            
    def _set_radio_button(self):
        values = self._get_radio_button_values()
        self.controller.set_radio_button(values)