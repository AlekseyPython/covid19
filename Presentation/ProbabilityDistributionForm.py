from .MixinQt import MixinQt
from .MixinButton import MixinButton
from .MixinParameters import MixinParameters
from .MixinSelections import MixinSelections

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinButton, MixinParameters, MixinSelections):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithTableAndParameter', title='Probability distribution of parameter')
        
        self.controller = controller
        MixinButton.__init__(self)
        MixinParameters.__init__(self)
        MixinSelections.__init__(self)
          
