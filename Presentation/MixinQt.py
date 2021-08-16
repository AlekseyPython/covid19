from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QWidget
from .MessageBox import MessageBox
from Entires.TypeMessage import TypeMessage
from Entires.Signal import Signal

class M_C(type(QWidget), type):pass

class MixinQt(QWidget, metaclass=M_C):
    messaged = QtCore.pyqtSignal(QtCore.QVariant)
    
    def __init__(self, controller, file_name, title):
        QWidget.__init__(self)
        self._load_file_ui(file_name)
        self._decorate_form(title)
        self.messaged.connect(self._perform_message, QtCore.Qt.BlockingQueuedConnection)
        self._set_signal_message(controller)
        
    def _load_file_ui(self, file_name):
        uic.loadUi('Presentation/Forms/' + file_name + '.ui', self)
    
    def _decorate_form(self, title):
        self.setWindowTitle(title)
        self.label.setText(title)
            
    @QtCore.pyqtSlot(QtCore.QVariant)
    def _perform_message(self, data):
        type_message = data['type_message']
        text = data['text']
        informative_text = data['informative_text']
        func_answer = data['func_answer']
    
        message_box = MessageBox()
        if type_message == TypeMessage.question:
            answer = message_box.answer_is_ok(text, informative_text)
            func_answer(answer)
        else:
            message_box.show_message(type_message, text, informative_text)
    
    def _set_signal_message(self, controller):
        signal = Signal(self.messaged)
        controller.set_signal_message(signal)  





