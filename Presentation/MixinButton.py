from .MessageBox import MessageBox
from Entires.TypeMessage import TypeMessage


class MixinButton:
    PROCEDURES_PERFORM_BEFORE_TASK = []
        
    def __init__(self):
        if hasattr(self, 'controller'):
            self._connect_button()
        
    def _connect_button(self):
        self.ButtonStart.clicked.connect(self._press_button_start)
    
    def register_procedure(self, proc): 
        self.PROCEDURES_PERFORM_BEFORE_TASK.append(proc)
    
    def _press_button_start(self):
        self.setEnabled(False)
    
        try:
            for proc in self.PROCEDURES_PERFORM_BEFORE_TASK:
                proc()
                
        except Exception as err:
            message_box = MessageBox()
            message_box.show_message(TypeMessage.critical, 'Form elements are filled incorrectly!', str(err))
            self.setEnabled(True)
            return
    
        self.controller.perform_business_task()
        self.setEnabled(True) 
        
        
