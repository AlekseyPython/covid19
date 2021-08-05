from .MixinQt import MixinQt
from .MixinButton import MixinButton
from .MixinParameters import MixinParameters
from .MixinRadioButton import MixinRadioButton

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinButton, MixinParameters, MixinRadioButton):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithRadioButtonAndParameter', title='Get list of values for columns')
        
        self.controller = controller
        MixinRadioButton.__init__(self)
        MixinButton.__init__(self)
        MixinParameters.__init__(self)