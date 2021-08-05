from . import Functions


class Selection:
    def __init__(self, column, compare_operation, value, common=False):
        self.column = column
        self.compare_operation = compare_operation
        self.value = value
        self.common = common
        
    def get_column(self):
        return self.column

    def get_compare_operation(self):
        return self.compare_operation
    
    def get_value(self):
        return self.value
    
    def get_common(self):
        return self.common
        
    def __str__(self):
        return self.column.get_name() +' ' + self.compare_operation + ' ' + str(self.value)
    
    
class Selections:
    def __init__(self):
        self.rows = []
    
    def add(self, selection):
        self.rows.append(selection)
        
    def add_value(self, column_name, value, compare_operation='=', converted=True):
        column = Functions.get_column_by_name(column_name, converted)
        selection = Selection(column, compare_operation, value)
        self.rows.append(selection)
    
    def get_columns(self):
        return list(set(map(Selection.get_column, self.rows)))
      
    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.rows):
            raise StopIteration
        else:
            selection = self.rows[self.current_index]
            self.current_index += 1
            return selection
        
    def __str__(self):
        seperator = ', '
        representations = map(str, self.rows)
        return seperator.join(representations)
