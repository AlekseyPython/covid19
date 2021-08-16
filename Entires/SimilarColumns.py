class SimilarColumns:
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns
        
    def get_name(self):
        return self.name
    
    def get_columns(self):
        return self.columns
    
    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.columns):
            raise StopIteration
        else:
            column = self.columns[self.current_index]
            self.current_index += 1
            return column
