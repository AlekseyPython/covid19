import copy
import numpy as np
import pandas as pd
from sklearn import model_selection
from Entires import SourceColumns as SC, ConvertedColumns as CC
from Entires import Functions, Enums
import Settings
from gpg.constants import data

class Features:
    def __init__(self, analysis_type, air_pollution):
        self.analysis_type = analysis_type
        
        self.use_converted_columns = True
        self.use_count_features = True
        self.use_air_pollutions = True
        
        self.all_features = []
        self.cat_features = []
        self._add_converted_features()
        self._add_counted_features()
        self._add_air_pollutions_features(air_pollution)
        self._add_geo_features()
        
        self.use_texts = False
        self.text_features = []
        self._add_text_features()
        
        #for repeating results, sort the columns
        self.all_features.sort()
        self.cat_features.sort()
        self.text_features.sort()
    
    def get_cat_features(self):
        return self.cat_features
        
    def get_text_features(self):
        return self.text_features
    
    def get_all_features(self):
        return list(set(self.all_features) | set(self.text_features))
    
    def _add_converted_features(self):
        if not self.use_converted_columns:
            return 
        
        names_types = self._get_names_types_converted_columns()
        names = [element[0] for element in names_types]
        self.all_features.extend(names)
        
        category_names = [element[0] for element in names_types if element[1] in ['bool', 'category']]
        self.cat_features.extend(category_names)
            
    def _get_names_types_converted_columns(self):
        names_types = []
        if not self.use_converted_columns:
            return names_types
        
        columns = []
        columns.append(CC.Age)
        columns.append(CC.Sex)
        columns.append(CC.GroupOfRisk)
        columns.append(CC.PeriodOfHospitalization)
        columns.append(CC.PeriodDiseaseForHospitalization)
        columns.append(CC.DIC)
        columns.append(CC.MV)
        columns.append(CC.AntiviralTreatment)
        columns.append(CC.TransferredToHospitalFromAnotherHospital)
        columns.append(CC.TransferredToHospitalFromQuarantine)
        columns.append(CC.TypeOfPneumonia)
        columns.append(CC.SaturationLevel)
        columns.append(CC.NameOfHospital)
        columns.append(CC.Source)
        columns.append(CC.StatusDecisionOfHospitalization)
        columns.append(CC.ResultOfHospitalization)
        columns.append(CC.TestInformation)
        columns.append(CC.Region)
        columns.append(CC.DidNotTravel)
        columns.append(CC.Country)
        columns.append(CC.ZodiacSign)     
        columns.append(CC.PrimaryElement)       
        columns.append(CC.NotFoundAtHome)
        columns.append(CC.PhoneOperator)
        columns.append(CC.ImmunosuppressantsDrugs)
        columns.append(CC.TreatmentHivInfectionDrugs)
        columns.append(CC.AntiviralDrugs)
        
        for column in columns:
            names_types.append((column.get_name(), column.get_type()))
        
        return names_types
        
    def _add_counted_features(self):
        if not self.use_count_features:
            return
        
        self._add_lenghts_comments_source_data()
        self._add_quantity_of_patients()
        self._add_rounded_features()
        self._add_quantities_days()
    
    def _add_lenghts_comments_source_data(self):
        self.all_features.append('lenght_comments_before_hospitalization')
        self.all_features.append('lenght_comment_of_hospitalization')
        self.all_features.append('lenght_comments_after_hospitalization')
    
    def _add_quantity_of_patients(self):
        self.all_features.append('quantity_patients_present_in_hospital')
        
    def _add_rounded_features(self):
        rounded_columns = []
        rounded_columns.append(CC.WeekDayArrivalAmbulance)
        rounded_columns.append(CC.WeekDayAdmissionToHospital)
        rounded_columns.append(CC.WeekDayDepartureFromHospital)
        rounded_columns.append(CC.Birthmonth)
        
        for coordinate in ['x', 'y']:
            for column in rounded_columns:
                name = column.get_name() + '_' + coordinate
                self.all_features.append(name)
    
    def _add_quantities_days(self):
        self.all_features.append('NumberDay')
        self.all_features.append('quantity_days_before_hospitalization')
        
    def _add_geo_features(self):
        if self.analysis_type == Enums.AnalysisTypes.ObtainingCharacteristicsOfSeriouslyDiseasePeople:
            return 
        
        self.all_features.append(CC.Longitude.get_name())
        self.all_features.append(CC.Latitude.get_name())
        self.all_features.append('radius_of_polar_coordinates')
        self.all_features.append('angle_of_polar_coordinates')
        
    def _add_text_features(self):
        if not self.use_texts:
            return
    
        for column in SC.all_comment:
            column_name = column.get_name()
            self.all_features.append(column_name)
            self.text_cfeatures.append(column_name)
    
    def _add_air_pollutions_features(self, air_pollution):
        for column in air_pollution.columns:
            if column == 'Сумма углеводородных соединений':
                #also have features ''Сумма углеводородных соединений за вычетом метана' and 'Метан', that correlate with this feature
                continue
            
            self.all_features.append(column)

    
class PreparedData:
    def __init__(self, analysis_type, features, source_data, converted_data, air_pollution):
        if len(converted_data) != len(source_data):
            raise RuntimeError('The count of source and converted data must be the same!')
        
        self.analysis_type = analysis_type
        self.features = features
        self.target_column = CC.SeverityOfDisease
        
        self.min_age = 16
        
        prepared_data = self._prepare_data(source_data, converted_data, air_pollution)
        self.X = self._count_X(prepared_data)
        self.Y = self._count_Y(prepared_data)
        
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def _count_X(self, prepared_data):
        X = pd.DataFrame(prepared_data, columns=self.features.all_features, copy=False)
        X = X.sort_index(axis=1)
        return X
    
    def _count_Y(self, prepared_data):
        target_name = self.target_column.get_name()
        return prepared_data[target_name].copy()
    
    def _prepare_data(self, source_data, converted_data, air_pollution):
        data = self._combine_converted_data_and_air_pollution(converted_data, air_pollution)
        data = self._add_comment_columns(data, source_data)
        data = self._selected_rows_data(data)
        data = self._convert_date_creating_to_number_day_from_start_registration(data)
        data = self._combine_group_of_risk_and_ecmo(data)
        data = self._add_death_to_target(data)
        data = self._remove_leaks_from_result_of_hospitalization(data)
        # data = self._lemmatize_text_columns(data)
        # data = self._remove_leaks_from_text_fields(data)
        data = self._convert_NA_values(data)
        data = self._count_counted_features(data)
        data = self._centred_geo_coordinates(data)
        data = self._count_polar_coordinates(data)
        return data
    
    def _combine_converted_data_and_air_pollution(self, converted_data, air_pollution): 
        air_pollution = air_pollution.astype('float32')
        all_data = pd.merge(converted_data, air_pollution, how='left', left_index=True, right_index=True)
        
        columns = list(air_pollution.columns)
        all_data[columns] = all_data[columns].fillna(value=-1)
        return all_data
            
    def _add_comment_columns(self, data, source_data):
        features = self.features
        if (not features.use_count_features) and (not features.use_texts):
            return data
        
        for column in SC.all_comment:
            column_name = column.get_name()
            series = pd.Series(source_data[column_name], name=column_name, dtype=column.get_type())
            data[column_name] = series
        return data
        
    def _selected_rows_data(self, data):
        data = data[(data.DecisionOfAmbulance=='Стационар') | (data.TransferredToHospitalFromQuarantine==True)]
        if self.features.use_converted_columns:
            data = data[data['Age'] >= self.min_age]
            data = data[data['Sex'].notnull()]
        
        target_name = self.target_column.get_name()
        data = data[data[target_name].notnull()]
        data = data.reset_index(drop=True)
        return data
    
    def _convert_date_creating_to_number_day_from_start_registration(self, data):
        data['NumberDay'] = data['DateCreating'] - data['DateCreating'].min()
        data['NumberDay'] /= np.timedelta64(1, 'D')
        data['NumberDay'] = data['NumberDay'].astype(np.int8)
        return data
    
    def _combine_group_of_risk_and_ecmo(self, data):
        if not self.features.use_converted_columns:
            return data
        
        categories, odered = CC.GroupOfRisk.get_categories()
        categories.append('ЭКМО')
        cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=odered)
        
        group_of_risk_name = CC.GroupOfRisk.get_name()
        ecmo_name = CC.ECMO.get_name()
        
        group_of_risk = data[group_of_risk_name]
        group_of_risk = group_of_risk.astype(cat_dtype)
        group_of_risk[data[ecmo_name]==True] = 'ЭКМО'
        data[group_of_risk_name] = group_of_risk
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
    
    def _remove_leaks_from_result_of_hospitalization(self, data):
        if not self.features.use_converted_columns:
            return data
        
        categories, odered = CC.ResultOfHospitalization.get_categories()
        categories.remove('Умер')
        cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=odered)
        
        column_name = CC.ResultOfHospitalization.get_name()
        series = data[column_name]
        series[series=='Умер'] = pd.NA
        series = series.astype(cat_dtype)
        data[column_name] = series
        return data
    
    # def _lemmatize_text_columns(self, data):
    #     features = self.features
    #     if (not features.use_count_features) and (not features.use_texts):
    #         return data
    #
    #     lemmatizator = Lemmatizator(data)
    #     data = lemmatizator.lemmatize(columns=features.get_text_features())
    #
    #     if self.use_words:
    #         all_texts = data['all_texts'].to_numpy()
    #         count = CountVectorizer(min_df=100, max_features=5000)
    #         bag = count.fit_transform(all_texts)
    # #         frequency_of_words = bag.toarray()
    #
    #         tf_idf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    #         bag = tf_idf.fit_transform(bag)
    #         frequency_of_words = bag.toarray()
    #         data = pd.concat([data, pd.DataFrame(frequency_of_words)], axis=1)
    #     return data
    #
    # def _remove_leaks_from_text_fields(self, data):
    #     features = self.features
    #     if (not features.use_count_features) and (not features.use_texts):
    #         return data
    #
    #     stop_words = ['симптомный', 'безсимптомный', 'легко', 'лёгкий', 'средний', 'средне', 'тяжёлый', 'тяжело',
    #                   'скончаться', 'смерть', 'умереть', 'умирать']
    #
    #     for column_name in features.get_text_features():
    #         for index in range(len(data)):
    #             sentence = data.loc[index, column_name]
    #             if not sentence:
    #                 continue
    #
    #             for word in stop_words:
    #                 sentence = sentence.replace(word, '')
    #             data.loc[index, column_name] = sentence
    #     return data

    def _convert_NA_values(self, data):
        if not self.features.use_converted_columns:
            return data
        
        converted_columns = Functions.get_columns(converted=True, with_names=True)
        category_columns = self.features.get_cat_features()
        for column_name in category_columns:
            column = converted_columns[column_name]
            if column.get_type() == 'bool':
                #bool values store as int with -1 as empy- value
                continue
            
            categories, odered = column.get_categories()
            categories.append('<NA>')
            cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=odered)
            
            series = data[column_name]
            series = series.astype(cat_dtype)
            series[series.isnull()] = '<NA>'
            data[column_name] = series
        return data 
    
    def _count_counted_features(self, data):
        if not self.features.use_count_features:
            return data
        
        data = self._count_quantities_days_before_hospitalization(data)
        data = self._count_lenghts_text_columns(data)
        data = self._count_quantity_of_patients(data)
        data = self._count_rounded_features(data)
        return data
    
    def _count_quantities_days_before_hospitalization(self, data):
        data['quantity_days_before_hospitalization'] = data[CC.DateAdmissionToHospital.get_name()] - data[CC.DateCreating.get_name()]
        data['quantity_days_before_hospitalization'] /= np.timedelta64(1, 'D')
        
        data['quantity_days_before_hospitalization'][data[CC.DateAdmissionToHospital.get_name()] == Settings.EMPTY_DATE] = 0
        data['quantity_days_before_hospitalization'][data[CC.DateCreating.get_name()] == Settings.EMPTY_DATE] = 0
        data['quantity_days_before_hospitalization'][data['quantity_days_before_hospitalization'] < 0] = 0
        data['quantity_days_before_hospitalization'] = data['quantity_days_before_hospitalization'].astype(np.int8)
        return data
     
    def _count_lenghts_text_columns(self, data):
        def name(column):
            return 'lenght_' + column.get_name()
        
        for column in SC.all_comment:
            data[name(column)] = data[column.get_name()].apply(len)
        
        data['lenght_comments_before_hospitalization'] = data[name(SC.CommentOfPhoneTalking)] + data[name(SC.CommentOfAmbulance)]
        data['lenght_comment_of_hospitalization']      = data[name(SC.CommentOfHospitalization)]
        data['lenght_comments_after_hospitalization']  = data[name(SC.CommentOfAftercare)] + data[name(SC.CommentOfQuarantine)]
        
        column_names = map(name, SC.all_comment)
        data = data.drop(column_names, axis=1)
        
        return data
    
    def _count_quantity_of_patients(self, data):
        def get_array_for_hospital_and_date(name_date, name_zeros, name_ones):
            data_hd = data[[name_of_hospital, name_date]].copy(deep=True)
            
            filled_mask = (data_hd[name_of_hospital]!='<NA>') & (data_hd[name_date]!=Settings.EMPTY_DATE)
            data_hd = data_hd[filled_mask]
            
            data_hd[name_zeros] = 0
            data_hd[name_ones] = 1
            data_hd = data_hd.rename(columns={name_date: 'Date'})
            return data_hd
        
        name_of_hospital = CC.NameOfHospital.get_name()
        name_admission = CC.DateAdmissionToHospital.get_name()
        name_departure = CC.DateDepartureFromHospital.get_name()
        
        data_admission = get_array_for_hospital_and_date(name_admission, name_zeros='departure_from_hospital', name_ones='admission_to_hospital')
        data_departure = get_array_for_hospital_and_date(name_departure, name_zeros='admission_to_hospital', name_ones='departure_from_hospital')
        data_balance = pd.concat([data_admission, data_departure], ignore_index=True)
        
        data_balance = data_balance.sort_values(by=['Date'])
        grouped_by_hospital = data_balance.groupby(data_balance[name_of_hospital])
        grouped_by_hospital = grouped_by_hospital.cumsum()
        
        data_balance = data_balance.drop(['admission_to_hospital', 'departure_from_hospital'], axis=1)
        data_balance = pd.merge(data_balance, grouped_by_hospital, how='left', left_index=True, right_index=True)
        
        #crate new index and drop duplicates
        data_balance = data_balance.set_index([name_of_hospital, 'Date'])
        data_balance = data_balance[~data_balance.index.duplicated(keep='last')]
        
        data_balance['present_in_hospital'] = data_balance['admission_to_hospital'] - data_balance['departure_from_hospital']
        data_balance = data_balance.drop(['admission_to_hospital', 'departure_from_hospital'], axis=1)
        
        grouped_balance = data_balance['present_in_hospital'].groupby(level=name_of_hospital)
        max_balance_for_hospitals = grouped_balance.max().dropna() #for categorical data, all values of the category are counted, even those for which there is no data, therefore dropna()
        
        #so that there are no duplicated names after the operation of the union of the tables
        # data_balance = data_balance.rename(columns={'balance': 'quantity_present'})
        max_balance_for_hospitals = max_balance_for_hospitals.rename("max_balance")
        
        #found counts on moment admission patien in hosptial
        name_moment = name_admission
        data_moment = data[[name_of_hospital, name_moment]].copy(deep=True)
        
        data_moment = pd.merge(data_moment, data_balance, how='left',  left_on=[name_of_hospital, name_moment], right_index=True)
        data_moment = pd.merge(data_moment, max_balance_for_hospitals, how='left',  left_on=[name_of_hospital], right_index=True)
        
        #for empty values, we didn't receive values of indicators
        empty_mask = (data_moment[name_of_hospital]=='<NA>') | (data_moment[name_moment]==Settings.EMPTY_DATE)
            
        data_moment['present_in_hospital'] /= data_moment['max_balance']
        data_moment['present_in_hospital'][empty_mask] = -1
        
        data_moment['present_in_hospital'] /= data_moment['present_in_hospital'].max()
        data_moment['present_in_hospital'][empty_mask] = -1
        
        name = 'quantity_patients_present_in_hospital'
        data[name] = pd.Series(data_moment['present_in_hospital'], name=name, dtype='float32')
        
        return data
    
    def _count_rounded_features(self, data):
        if not self.features.use_converted_columns:
            return
        
        replaced_features = []
        all_features = self.features.get_all_features()
        for feature_name_with_postfix in all_features:
            exist_x_postfix = feature_name_with_postfix.endswith('_x')
            exist_y_postfix = feature_name_with_postfix.endswith('_y')
            if not exist_x_postfix and not exist_y_postfix:
                continue
            
            feature_name = feature_name_with_postfix[:-2]
            feature_data = data[feature_name].to_numpy()
            quantity = feature_data.max()
            if exist_x_postfix:
                coordinates = np.cos(feature_data * 2*np.pi/quantity)
            else:
                coordinates = np.sin(feature_data * 2*np.pi/quantity)
            
            coordinates[feature_data==-1] = -2
            coordinates = pd.Series(coordinates, name=feature_name_with_postfix, dtype='float32')
               
            data[feature_name_with_postfix] = coordinates
            replaced_features.append(feature_name)
        
        replaced_features = list(set(replaced_features))    
        data = data.drop(replaced_features, axis=1)      
        return data
    
    def _centred_geo_coordinates(self, data):
        if self.analysis_type == Enums.AnalysisTypes.ObtainingCharacteristicsOfSeriouslyDiseasePeople:
            return 
        
        for feature, center in zip(['Longitude', 'Latitude'], Settings.CENTER_OF_MOSCOW):
            source_coordinate = data[feature].to_numpy()
            coordinate = source_coordinate - center
            coordinate[source_coordinate==-1] = -1
            coordinate = pd.Series(coordinate, name=feature, dtype='float32')
            
            data[feature] = coordinate
        return data
    
    def _count_polar_coordinates(self, data):
        if self.analysis_type == Enums.AnalysisTypes.ObtainingCharacteristicsOfSeriouslyDiseasePeople:
            return 
        
        x = data['Longitude'].to_numpy()
        y = data['Latitude'].to_numpy()
        
        #radius
        radius = np.sqrt(np.float_power(x, 2) + np.float_power(y, 2))
        radius *= 1 / radius.max()
        radius[np.isnan(radius)] = -1
        radius = pd.Series(radius, name='radius_of_polar_coordinates', dtype='float32')
        radius[x==-1] = -1
        
        #angle
        angle = np.arctan2(x, y) #[-pi; pi]
        angle = angle + np.pi
        angle[np.isnan(angle)] = -4
        angle = pd.Series(angle, name='angle_of_polar_coordinates', dtype='float32')
        angle[x==-1] = -1
        
        data['radius_of_polar_coordinates'] = radius
        data['angle_of_polar_coordinates'] = angle
        return data 
    
    
class LearningData:
    def __init__(self, prepared_data, features, severity_of_disease, random_state=42, train_size=0.8, excluded_features=[]):
        self.prepared_data = prepared_data
        self.features = features
        self.severity_of_disease = severity_of_disease
        
        #ignored_features in CatBoostClassifier and Pool doesn't work, therefore modifing LearningData
        self.excluded_features = self._get_all_excluded_features(excluded_features, severity_of_disease)
        self.random_state = random_state
        
        self.shuffled_feature = None
        
        X = self._count_X()
        Y = self._count_Y()
        if train_size == 0:
            #empty value
            train_size=0.8
            
        self._train_test_split(X, Y, train_size)
    
    def profile_report(self, silently):
        def get_sample(X):
            if Settings.debuge_mode:
                np.random.seed(42)
            
            random_state = np.random.randint(low=0, high=10000, size=1)
            return X.sample(n=1000, random_state=random_state)
            
        #very long import, therefore do it in function
        from pandas_profiling import ProfileReport
        
        X = self._count_X()
        if Settings.debuge_mode and len(X) > 1000:
            X = get_sample(X)
            
        profile = ProfileReport(X, 
                                dark_mode=True, 
                                correlations={'pearson':    {'calculate': True}, #other correlations don't carry additional useful information 
                                              'spearman':   {'calculate': False},
                                              'kendall':    {'calculate': False},
                                              'phi_k':      {'calculate': False},
                                              'cramers':    {'calculate': False},
                                              'recoded':    {'calculate': False}},
                                duplicates  ={'head':       0}) #Error pandas_profiling!!!
        
        file_name = Settings.FOLDER_FOR_TEMPORARY_FILES + '/Profile of data for severity of disease = ' + self.severity_of_disease + '.html'
        return profile.to_file(output_file=file_name, silent=silently)
        
    def shuffle(self, feature_name): 
        if self.shuffled_feature is None:
            self.shuffled_feature = feature_name
            self.feature_x_train = self.X_train[feature_name].copy()
            self.feature_x_test = self.X_test[feature_name].copy()
            
        else:    
            if self.shuffled_feature != feature_name:
                raise RuntimeError('First restore the previous shuffled feature!')
            
        for _ in range(3):
            self.X_train[feature_name] = self.X_train[feature_name].sample(frac=1, random_state=self.random_state).values
            self.X_test[feature_name] = self.X_test[feature_name].sample(frac=1, random_state=self.random_state).values
        
    def unshuffle(self, feature_name):
        if self.shuffled_feature is None or self.shuffled_feature != feature_name:
            raise RuntimeError("This feature wasn't shuffled!")
        
        self.X_train[feature_name] = self.feature_x_train
        self.X_test[feature_name] = self.feature_x_test
        self.shuffled_feature = None
        
    def get_train_samples(self):
        return self.X_train, self.Y_train
    
    def get_test_samples(self):
        return self.X_test, self.Y_test
    
    def get_current_features(self):
        current_features = copy.deepcopy(self.features)
        current_features.all_features = list(set(current_features.all_features) - set(self.excluded_features))
        current_features.cat_features = list(set(current_features.cat_features) - set(self.excluded_features))
        current_features.text_features = list(set(current_features.text_features) - set(self.excluded_features))
        return current_features
        
    def _get_all_excluded_features(self, excluded_features, severity_of_disease):
        if severity_of_disease == 'Умер':
            additional_names = []
            additional_names.append(CC.ResultOfHospitalization.get_name())
            additional_names.append(CC.WeekDayDepartureFromHospital.get_name() + '_x')
            additional_names.append(CC.WeekDayDepartureFromHospital.get_name() + '_y')
            additional_names.append('lenght_comments_after_hospitalization')
            
            excluded_features = list(set(excluded_features) | set(additional_names))
        return excluded_features
            
    def _count_X(self):
        current_X = self.prepared_data.get_X().copy(deep=True)
        if self.excluded_features:
            current_X = current_X.drop(self.excluded_features, axis=1)
        return current_X
                
    def _count_Y(self):
        Y = self.prepared_data.get_Y()
        array = np.zeros(len(Y), dtype='bool')
        
        target_name = self.prepared_data.target_column.get_name()
        current_Y = pd.Series(array, name=target_name, dtype='bool')
        current_Y[Y>=self.severity_of_disease] = True
        return current_Y
        
    def _train_test_split(self, X, Y, train_size):
        if train_size<=0 or train_size>=1:
            raise RuntimeError('The part of the training sample should be in the range (0; 1)')
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(X, Y, train_size=train_size, random_state=self.random_state, stratify=Y)
            
    

   
