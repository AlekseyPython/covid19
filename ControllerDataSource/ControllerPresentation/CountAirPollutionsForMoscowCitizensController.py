from .AController import AController
from Entires.TypeMessage import TypeMessage
import Initialization


class Controller(AController):
    def __init__(self):
        AController.__init__(self)
    
    def get_radio_buttons(self):
        return ['Without stations located near the roads', 'With stations located near the roads']
    
    def set_radio_button(self, values):
        self.include_stations_near_the_roads = values['With stations located near the roads']
        
    def perform_task(self):
        return Initialization.ibusiness.count_air_pollutions_for_moscow_citizens(self.signal_message, self.include_stations_near_the_roads)
    
    def show_result(self, result):
        if result['data']:
            text = 'Operation successfully complete!'
            ipresentation = Initialization.ipresentation
            ipresentation.message_to_user(TypeMessage.information, text)