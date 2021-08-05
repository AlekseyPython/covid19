from .MixinQt import MixinQt
from .MixinButton import MixinButton
from .MixinParameters import MixinParameters
from .MixinRadioButton import MixinRadioButton

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinButton, MixinParameters, MixinRadioButton):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithFourParameters', title='Tuning a prediction model for disease severity')
        
        self.controller = controller
        MixinButton.__init__(self)
        MixinParameters.__init__(self)
        MixinRadioButton.__init__(self)

