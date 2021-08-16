from .MixinQt import MixinQt
from .MixinButton import MixinButton
from .MixinRadioButton import MixinRadioButton

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinRadioButton, MixinButton):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithRadioButton', title='Count air pollutions for Moscow citizens')

        self.controller = controller
        MixinRadioButton.__init__(self)
        MixinButton.__init__(self)
        