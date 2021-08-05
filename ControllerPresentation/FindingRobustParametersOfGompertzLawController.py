from .AController import AController
from Entires.TypeMessage import TypeMessage
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
        self.description = 'Finding robust parameters of the Gompertz law'
    
    def get_description(self):
        return self.description
    
    def perform_task(self):
        return Initialization.ibusiness.convert_data_to_desired_types(self.signal_message)
    
    def show_result(self, result):
        if result['data']:
            text = 'Operation successfully complete!'
            ipresentation = Initialization.ipresentation
            ipresentation.message_to_user(TypeMessage.information, text)
