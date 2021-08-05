from .MixinQt import MixinQt
from .MixinButton import MixinButton

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinButton):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithButton', title='Creater separated maps of Moscow regions')

        self.controller = controller
        MixinButton.__init__(self)