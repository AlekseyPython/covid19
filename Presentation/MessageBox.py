from PyQt5.QtWidgets import QMessageBox
from Entires.TypeMessage import TypeMessage


class MessageBox(QMessageBox):
    def __init__(self):
        QMessageBox.__init__(self)
    
    def answer_is_ok(self, text, informative_text=''):
        self.setText(text)
        self.setInformativeText(informative_text)
        self.setIcon(QMessageBox.Question) 
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.setDefaultButton(QMessageBox.Cancel)
        
        result = self.exec_()
        return result == QMessageBox.Ok
    
    def show_message(self, type_message, text, informative_text=''):
        self.setText(text)
        self.setInformativeText(informative_text)
        
        if type_message == TypeMessage.information:
            icon = QMessageBox.Information
        elif type_message == TypeMessage.warning:
            icon = QMessageBox.Warning
        elif type_message == TypeMessage.critical:
            icon = QMessageBox.Critical
        else:
            icon = QMessageBox.NoIcon
        self.setIcon(icon) 
           
        self.setStandardButtons(QMessageBox.Ok)
        self.setDefaultButton(QMessageBox.Ok)
        self.exec_()