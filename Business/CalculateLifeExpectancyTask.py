import pandas as pd
from .ATask import ATask
from Entires import ConvertedColumns
from . import Functions
import Initialization


class Task(ATask):
    def __init__(self, signal_message, parameter, selections):
        ATask.__init__(self)
        
        self.signal_message = signal_message
        self.parameter = parameter
        self.selections = selections
    
    def run(self):pass
        
            
    
        
