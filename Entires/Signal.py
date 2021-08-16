import Settings


class Signal:
    def __init__(self, signal):
        self.signal_message = signal
        
    def emit(self, data:dict):
        if Settings.debuge_mode:
            if 'func_answer' in data:
                data['func_answer'](True)
            else:
                raise RuntimeError(data['text'])
        else:
            self.signal_message.emit(data)
        
        
    
            
