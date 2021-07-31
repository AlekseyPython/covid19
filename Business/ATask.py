from abc import ABCMeta
import Settings
 
 
if Settings.debuge_mode:
    from threading import Thread
    M_C = ABCMeta
else:  
    from PyQt5.QtCore import QThread as Thread  
    class M_C(type(Thread), ABCMeta):pass

class ATask(Thread):
    __metaclass__ = M_C
    def __init__(self):
        Thread.__init__(self)
        self.result = None
    
    
        
        

    