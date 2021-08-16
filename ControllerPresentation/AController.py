from abc import ABCMeta, abstractmethod
import Initialization
from Entires.TypeMessage import TypeMessage
from Entires import Functions

class AController(metaclass=ABCMeta):
    def __init__(self):pass
    
    def get_columns_for_selection(self):
        return Functions.get_columns(converted=True)
    
    def set_signal_message(self, signal_message):
        self.signal_message = signal_message
    
    def perform_business_task(self):
        result = self.perform_task()
        if result is None:
            Initialization.ipresentation.message_to_user(TypeMessage.critical, 'The task has not yet been completed!')
            return
        
        window = self.show_result(result)
        if not hasattr(self, 'windows'):
            self.windows = []
        self.windows.append(window)
    
    @abstractmethod        
    def perform_task(self): pass

    @abstractmethod
    def show_result(self, result): pass
