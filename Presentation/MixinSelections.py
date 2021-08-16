import pandas as pd
import numpy as np
import time
from PyQt5 import QtWidgets
from PyQt5.QtCore import QVariant
from .Delegates import SpinBoxDelegate, DateDelegate
from Entires.TypeMessage import TypeMessage
from Entires.Selections import Selection, Selections
import Initialization


class MixinSelections:
    def __init__(self):
        if not hasattr(self, 'controller'):
            return 
        
        self._set_names()
        self._connect_add_delete_buttons()
        
        self.columns_table = []
        self.columns_table.append({'name':'Column', 'width':300, 'cell_pressed': self._column_cell_pressed, 'get_value':self._column_get_value})
        self.columns_table.append({'name':'Compare operation', 'width':150, 'cell_pressed': self._compare_cell_pressed, 'get_value':self._compare_get_value})
        self.columns_table.append({'name':'Value', 'width':400, 'cell_pressed': self._value_cell_pressed, 'get_value':self._value_get_value})
        self._fill_table()
        
        if hasattr(self, 'register_procedure'):
            self.register_procedure(self._set_selections)
    
    def _set_names(self):
        self.LabelTable.setText('Selections of source data:')
        
    def _connect_add_delete_buttons(self):
        self.ButtonAddItem.clicked.connect(self._press_button_add_item)
        self.ButtonDeleteItem.clicked.connect(self._press_button_delete_item)
    
    def _press_button_add_item(self):
        row = self.TableTable.rowCount()
        self.TableTable.insertRow(row)

    def _press_button_delete_item(self):
        self.time_press_button_delete = time.time()
        row = self.TableTable.currentRow()
        self.TableTable.removeRow(row)
        
    def _set_selections(self):
        self.controller.set_selections(self.get_selections())
        
    def _fill_table(self):
        self.TableTable.clear()
        self.TableTable.setColumnCount(len(self.columns_table))
        
        names = [dic['name'] for dic in self.columns_table]
        self.TableTable.setHorizontalHeaderLabels(names)
        
        for index, dic in enumerate(self.columns_table):
            self.TableTable.setColumnWidth(index, dic['width'])
            
        self.TableTable.cellPressed.connect(self._current_cell_pressed)
        self.TableTable.currentCellChanged.connect(self.currentCellChanged)
        
    def currentCellChanged(self, currentRow, currentColumn, previousRow, previousColumn):
        if hasattr(self, 'time_press_button_delete'):
            if time.time() - self.time_press_button_delete < 0.1:
                return
        
        self._current_cell_pressed(currentRow, currentColumn)

    def _current_cell_pressed(self, index_row, index_column):
        func = self.columns_table[index_column]['cell_pressed']
        func(index_row, index_column)
    
    def _column_cell_pressed(self, index_row, index_column):
        comboBoxColumn = QtWidgets.QComboBox()
        columns = self.controller.get_columns_for_selection()
        for column in columns:
            comboBoxColumn.addItem(column.get_name(), QVariant(column))
        self.TableTable.setCellWidget(index_row, index_column, comboBoxColumn)
        
    def _get_value_column(self, index_row):
        index_column_with_value_column = [index for index, dic in enumerate(self.columns_table) if dic['name']=='Column'][0]
        widget = self.TableTable.cellWidget(index_row, index_column_with_value_column)
        if widget is None:
            return None
        else:
            return widget.currentData()
        
    def _compare_cell_pressed(self, index_row, index_column):
        value_column = self._get_value_column(index_row)
        if value_column is None:
            return 
        
        compare_operations = value_column.get_compare_operations()
        comboBoxCompare = QtWidgets.QComboBox()
        comboBoxCompare.addItems(compare_operations)
        self.TableTable.setCellWidget(index_row, index_column, comboBoxCompare)
        
    def _value_cell_pressed(self, index_row, index_column):
        value_column = self._get_value_column(index_row)
        if value_column is None:
            return
        
        self.TableTable.removeCellWidget(index_row, index_column)
        if value_column.numeric_type():
            delegate = SpinBoxDelegate()
        elif value_column.date_type():
            delegate = DateDelegate()
        else:
            delegate = QtWidgets.QItemDelegate()

            cell_widget = QtWidgets.QComboBox()
            cell_widget.clear()
            
            possible_values = value_column.get_possible_values()
            cell_widget.addItems(possible_values)
            self.TableTable.setCellWidget(index_row, index_column, cell_widget)
             
        self.TableTable.setItemDelegateForColumn(index_column, delegate)
        
    def get_selections(self):
        selections = Selections()
        rows = self.TableTable.rowCount()
        for index_row in range(rows):
            selection = {}
            for index_column, dic_column in enumerate(self.columns_table):
                name = dic_column['name'].lower()
                name = name.replace(' ', '_')
                
                selection[name] = dic_column['get_value'](index_row, index_column)
            
            selection = Selection(**selection)    
            selections.add(selection)
        return selections
        
    def _column_get_value(self, index_row, index_column):
        column = self.TableTable.cellWidget(index_row, index_column)
        if column is None:
            Initialization.ipresentation.message_to_user(TypeMessage.critical, 'Please, fill the selection table correctly!')
            raise RuntimeError()
        return column.currentData()
    
    def _compare_get_value(self, index_row, index_column):
        compare_operation = self.TableTable.cellWidget(index_row, index_column)
        if compare_operation is None:
            Initialization.ipresentation.message_to_user(TypeMessage.critical, 'Please, fill the selection table correctly!')
            raise RuntimeError()
        return compare_operation.currentText()
    
    def _value_get_value(self, index_row, index_column):
        index_column_with_value_column = [index for index, dic in enumerate(self.columns_table) if dic['name']=='Column'][0]
        data_column = self._column_get_value(index_row, index_column_with_value_column)
        if data_column.numeric_type() or data_column.date_type():
            value = self.TableTable.item(index_row, index_column)
        else:
            value = self.TableTable.cellWidget(index_row, index_column)
            
        ipresentation = Initialization.ipresentation
        if value is None:
            ipresentation.message_to_user(TypeMessage.critical, 'Please, fill the selection table correctly!')
            raise RuntimeError()
        
        if data_column.numeric_type():
            value = value.text()
            if not value:
                raise RuntimeError('Selection table is filled incorrectly!')
            
            return int(value)
                
        elif data_column.date_type():
            value = value.text()
            if not value:
                raise RuntimeError('Selection table is filled incorrectly!')
            
            return np.datetime64(value, 'D')
                
        elif data_column.bool_type():
            value = value.currentText()
            if not value:
                raise RuntimeError('Selection table is filled incorrectly!')
            
            if value == 'False':
                return False
            elif value == 'True':
                return True
            else:
                return pd.NA
        
        else:
            value = value.currentText()
            if not value:
                raise RuntimeError('Selection table is filled incorrectly!')
            
            if value == '<NA>':
                return pd.NA
            return value
        
        