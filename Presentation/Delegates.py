from PyQt5 import QtCore, QtWidgets


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self):
        QtWidgets.QStyledItemDelegate.__init__(self)
        
    def createEditor(self, parent, model, index):
        editor = QtWidgets.QSpinBox(parent)
        editor.setFrame(False)
        editor.setMinimum(-1)
        editor.setMaximum(2000)
        editor.setSingleStep(1)
        return editor
    
    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        if not value:
            value = -1
        else:
            value = int(value)
        
        if type(value) == int:
            editor.setValue(value)
        
    def updateEditorGeometry(self, editor, options, index):
        editor.setGeometry(options.rect)
        
    def setModelData(self, editor, model, index):
        value = editor.value()
        model.setData(index, value, QtCore.Qt.EditRole)
        
        
class DateDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self):
        QtWidgets.QStyledItemDelegate.__init__(self)
        
    def createEditor(self, parent, option, index):
        editor = QtWidgets.QDateEdit(parent)
        editor.setMaximumDate(QtCore.QDate(2020, 5, 4))
        editor.setFrame(False)
        return editor
    
    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        if type(value) == QtCore.QDate:
            editor.setDate(value)
        
    def updateEditorGeometry(self, editor, options, index):
        editor.setGeometry(options.rect)
        
    def setModelData(self, editor, model, index):
        value = editor.date()
        model.setData(index, value, QtCore.Qt.EditRole)
        
        
        