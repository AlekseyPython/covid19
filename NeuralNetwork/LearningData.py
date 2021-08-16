import multiprocessing
import numpy as np
import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset
from Entires import SourceColumns as SC, ConvertedColumns as CC
from .Lemmatizator import Lemmatizator


class PreparedData:
    def __init__(self, source_data, converted_data):
        if len(converted_data) != len(source_data):
            raise RuntimeError('The count of source and converted data must be the same!')
        
        self.target_column = CC.SeverityOfDisease
        self.min_age = 16
        self.feature_names = list(map(SC._Column.get_name, SC.all_comment))
        
        prepared_data = self._prepare_data(source_data, converted_data)
        self.X = self._count_X(prepared_data)
        self.Y = self._count_Y(prepared_data)
        
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def _count_X(self, prepared_data):
        X = pd.DataFrame(prepared_data, columns=self.feature_names, copy=False)
        X = X.sort_index(axis=1)
        return X
    
    def _count_Y(self, prepared_data):
        target_name = self.target_column.get_name()
        return prepared_data[target_name].copy()
    
    def _prepare_data(self, source_data, converted_data):
        data = self._add_comment_columns(converted_data, source_data)
        data = self._selected_rows_data(data)
        data = self._add_death_to_target(data)
        data = self.remove_unnecessary_columns(data) 
        data = self._lemmatize_text_columns(data)
        data = self._remove_leaks_from_text_fields(data)
        return data
    
    def _add_comment_columns(self, data, source_data):
        for column in SC.all_comment:
            column_name = column.get_name()
            series = pd.Series(source_data[column_name], name=column_name, dtype=column.get_type())
            data[column_name] = series
        return data
        
    def _selected_rows_data(self, data):
        data = data[(data.DecisionOfAmbulance=='Стационар') | (data.TransferredToHospitalFromQuarantine==True)]
        data = data[data['Age'] >= self.min_age]
        data = data[data['Sex'].notnull()]
        
        target_name = self.target_column.get_name()
        data = data[data[target_name].notnull()]
        data = data.reset_index(drop=True)
        return data
    
    def _add_death_to_target(self, data):
        target_name = self.target_column.get_name()
        categories, odered = self.target_column.get_categories()
        categories.append('Умер')
        cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=odered)
        
        target = data[target_name]
        target = target.astype(cat_dtype)
        target[data['Death']==True] = 'Умер'
        data[target_name] = target
        return data
    
    def remove_unnecessary_columns(self, data):
        removing_columns = []
        name_target_column = self.target_column.get_name()
        for column in data.columns:
            if column in self.feature_names or column==name_target_column:
                continue
            
            removing_columns.append(column)
        data = data.drop(removing_columns, axis=1)
        return data
         
    def _lemmatize_text_columns(self, data):
        lemmatizator = Lemmatizator(remove_end_punctuation_sentences=True)
        return lemmatizator.lemmatize_data(data, columns=self.feature_names)
    
    def _remove_leaks_from_text_fields(self, data):
        for column_name in self.feature_names:
            data_column = data[column_name]
            with multiprocessing.Pool() as pool:
                data_column_without_leaks = pool.map(self._remove_leaks, data_column)
                
            data[column_name] = data_column_without_leaks
        return data
    
    @staticmethod
    def _remove_leaks(sentence):
        if not sentence:
            return sentence
        
        stop_words = ('симптомный', 'безсимптомный', 'легко', 'лёгкий', 'средний', 'средне',
                  'тяжёлый', 'тяжело', 'скончаться', 'смерть', 'умереть', 'умирать')
        
        for word in stop_words:
            sentence = sentence.replace(word, '')
        return sentence


class LearningData(Dataset):
    def __init__(self, prepared_data, severity_of_disease, random_state=42, train_size=0.8, train=False):
        #Dataset.__init__(self) __init__ not determined
        self.prepared_data       = prepared_data
        self.severity_of_disease = severity_of_disease
        self.random_state        = random_state
        self.train_size          = train_size
        self.train               = train
        
        self.excluded_features = self._get_excluded_features()
        
        X = self._count_X()
        Y = self._count_Y()
        
        self.X, self.Y = self._train_test_split(X,Y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        common_data = ''
        separator   = ' '
        series      = self.X.iloc[idx]
        for column in self.X.columns:
            common_data += separator + series[column]
        
        return common_data, self.Y.iloc[idx]
    
    def _get_excluded_features(self): 
        excluded_features = []  
        if self.severity_of_disease == 'Умер':
            excluded_features.append(SC.CommentOfQuarantine.get_name())
            excluded_features.append(SC.CommentOfAftercare.get_name())
        return excluded_features
             
    def _count_X(self):
        current_X = self.prepared_data.get_X().copy(deep=True)
        if self.excluded_features:
            current_X = current_X.drop(self.excluded_features, axis=1)
        return current_X
                
    def _count_Y(self):
        Y = self.prepared_data.get_Y()
        array = np.zeros(len(Y), dtype='float32')
        
        target_name = self.prepared_data.target_column.get_name()
        current_Y = pd.Series(array, name=target_name, dtype='float32')
        current_Y[Y>=self.severity_of_disease] = 1.0
        return current_Y
    
    def _train_test_split(self, X, Y):
        if self.train_size<=0 or self.train_size>=1:
            raise RuntimeError('The part of the training sample should be in the range (0; 1)')
        
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=self.train_size, random_state=self.random_state, stratify=Y)
        if self.train:
            return X_train, Y_train
        else:
            return  X_test, Y_test
        
        
        

   