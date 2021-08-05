import os
import functools
from Entires.TypeMessage import TypeMessage


class FilesChecker:
    def __init__(self, signal_message):
        self.signal_message = signal_message
    
    @staticmethod    
    def _get_answer(answer, file):
        if answer == True:
            os.remove(file)
        
    def absence_or_deleted(self, file, informational_text):
        if not os.path.exists(file):
            return True
        
        data = {}
        data['type_message'] = TypeMessage.question
        data['text'] = informational_text
        data['informative_text'] = 'Do you want to delete one and proceed?'
        data['func_answer'] = functools.partial(self._get_answer, file=file)
        self.signal_message.emit(data)
        
        if os.path.exists(file):
            return False
        else:
            return True
        
    def existence(self, file):
        if os.path.exists(file):
            return True
        
        data = {}
        data['type_message'] = TypeMessage.critical
        data['text'] = "The file doesn't exist: " + file
        self.signal_message.emit(data)
        return False
    
