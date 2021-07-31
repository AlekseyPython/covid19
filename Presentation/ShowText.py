from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget

class ShowText(QWidget):
    def __init__(self, text, title):
        QWidget.__init__(self)
        self.text = text
        self.title = title
    
    def show_window(self):
        self.setWindowTitle("List of common values for " + self.title)
        self.resize(500, 250)
        
        textEdit = QtWidgets.QTextEdit()
        textEdit.append(self.text)
        
        box = QtWidgets.QVBoxLayout()
        box.addWidget(textEdit)
        self.setLayout(box)
        
        self.show()  

