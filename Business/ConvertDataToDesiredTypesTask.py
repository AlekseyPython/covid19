from .ATask import ATask
import Initialization


class Task(ATask):
    def __init__(self, signal_message):
        ATask.__init__(self)
        self.signal_message = signal_message
    
    def run(self):
        self.result = {}
        self.result['data'] = Initialization.icontroller_data_sourse.convert_data(self.signal_message)
         
    