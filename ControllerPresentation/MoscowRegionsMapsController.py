from .AController import AController
from Entires.TypeMessage import TypeMessage
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def perform_task(self):
        return Initialization.ibusiness.create_moscow_regions_maps(self.signal_message)
    
    def show_result(self, result):
        if result['data']:
            text = 'Operation successfully complete!'
            ipresentation = Initialization.ipresentation
            ipresentation.message_to_user(TypeMessage.information, text)