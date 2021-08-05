from PyQt5 import QtWidgets
from .MixinQt import MixinQt
from .MixinButton import MixinButton
from .MixinParameters import MixinParameters
from .MixinSelections import MixinSelections

class M_C(type(MixinQt), type):pass

class Form(MixinQt, MixinButton, MixinParameters, MixinSelections):
    __metaclass__ = M_C
    def __init__(self, controller):
        MixinQt.__init__(self, controller, file_name='FormWithTableAndTwoParameters', title='Air pollution counted for Moscow citizens')
       
        self.controller = controller
        MixinButton.__init__(self)
        MixinParameters.__init__(self)
        MixinSelections.__init__(self)
        
        column_common ={'name':'Common', 'width':100, 'cell_pressed': self._common_cell_pressed, 'get_value':self._common_get_value}
        self.columns_table.insert(0, column_common)
        self._fill_table()
    
    def _common_cell_pressed(self, index_row, index_column):
        check_box_item = QtWidgets.QCheckBox()
        self.TableTable.setCellWidget(index_row, index_column, check_box_item)
            
    def _common_get_value(self, index_row, index_column):
        common = self.TableTable.cellWidget(index_row, index_column)
        if common is None:
            return False
        return common.isChecked()
    