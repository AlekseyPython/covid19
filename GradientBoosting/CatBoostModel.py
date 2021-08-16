from catboost import CatBoostClassifier, Pool, EFstrType, EFeaturesSelectionAlgorithm, EShapCalcType


class Pools:
    def __init__(self, features, learning_data):
        self.features = features
        self.learning_data = learning_data
        
        self.train_pool = self._count_train_pool()
        self.test_pool = self._count_test_pool()
    
    def get_train_pool(self):
        return self.train_pool
    
    def get_test_pool(self):
        return self.test_pool
       
    def _count_train_pool(self):
        X_train, Y_train = self.learning_data.get_train_samples()
        cat_features = self.features.get_cat_features()
        text_features = self.features.get_text_features()
        pool = Pool(X_train, Y_train, cat_features, text_features)
        # pool.quantize(border_count=512)#pool.quantize(self.ignored_features) - got an Error!
        return pool
    
    def _count_test_pool(self):
        X_test, Y_test = self.learning_data.get_test_samples()
        cat_features = self.features.get_cat_features()
        text_features = self.features.get_text_features()
        pool = Pool(X_test, Y_test, cat_features, text_features)
        # pool.quantize(border_count=512)#pool.quantize(self.ignored_features) - got an Error!
        return pool

        
class CatBoostModel:
    def __init__(self, features, pools, random_state=42, params_model={}, penalties={}, common_penalty=0):
        self.features = features
        self.pools = pools
        self.random_state = random_state
        self.text_processing_json = self._get_text_processing_json()
        
        self.params_model = self._get_common_parameters()
        self._set_parameters(params_model)
        
        if common_penalty > 0: 
            self._set_common_penalties(common_penalty)
        self._set_penalties(penalties)
        
        self.model = CatBoostClassifier()
        
    def _get_common_parameters(self): 
        common_params = {}
        common_params['iterations'] = 10 * 1000
        common_params['learning_rate'] = 0.05
        common_params['border_count'] = 512 #names of features are available only after fit, so it's impossible to set separate values for features 
        common_params['random_seed'] = self.random_state
        common_params['text_processing'] = self.text_processing_json
        common_params['use_best_model'] = True
        common_params['one_hot_max_size'] = 1
        common_params['early_stopping_rounds'] = 10
        common_params['l2_leaf_reg'] = 7
        common_params['boosting_type'] = 'Ordered'
        common_params['depth'] = 4
        #common_params['task_type'] = 'GPU'
        return common_params    
        
    def set_parameter(self, name, value):
        if value is None:
            return 
        
        params_model = self.params_model
        if name in params_model and type(params_model[name])==list:
            current_value = params_model[name]
            params_model[name] = list(set(current_value) | set(value))
        else:
            params_model[name] = value
            
    def set_feature_penalty(self, feature_name,  penalty):
        if 'per_object_feature_penalties' not in self.params_model:
            self.params_model['per_object_feature_penalties'] = {}
            self.params_model['first_feature_use_penalties'] = {}
            
        self.params_model['per_object_feature_penalties'][feature_name] = penalty
        self.params_model['first_feature_use_penalties'][feature_name] = penalty
    
    def _set_silently(self, silently):
        self.params_model['verbose'] = False if silently else 100
        
    def fit(self, silently=True):
        self._set_silently(silently)
        self.model.set_params(**self.params_model)
        
        train_pool = self.pools.get_train_pool()
        test_pool = self.pools.get_test_pool()
        
        params_fitting = {}
        params_fitting['X'] = train_pool
        params_fitting['eval_set'] = test_pool
        params_fitting['verbose'] = False if silently else 100
        self.model.fit(**params_fitting)
    
    def select_features(self, num_features_to_eliminate, silently=True):
        #Error with penalties!!!
        self._set_silently(silently)
        self.model.set_params(**self.params_model)
        
        train_pool = self.pools.get_train_pool()
        test_pool = self.pools.get_test_pool()
        
        params = {}
        params['X'] = train_pool
        params['eval_set'] = test_pool
        params['features_for_select'] = list(range(train_pool.shape[1]))
        params['num_features_to_select'] = train_pool.shape[1] - num_features_to_eliminate
        params['steps'] = min(3, num_features_to_eliminate) #to avoid warnings about automatic decrease in the number of steps 
        params['algorithm'] = EFeaturesSelectionAlgorithm.RecursiveByShapValues
        params['shap_calc_type'] = EShapCalcType.Regular
        params['train_final_model'] = True
        params['logging_level'] = 'Silent' if silently else 'Info'
        return self.model.select_features(**params)
        
    def get_predictions(self):
        if not self.model.is_fitted():
            raise RuntimeError('First fitting model!')
        
        train_pool = self.pools.get_train_pool()
        train_predictions = self.model.predict_proba(train_pool)[:,1]
            
        test_pool = self.pools.get_test_pool()
        test_predictions = self.model.predict_proba(test_pool)[:,1]
        return train_predictions, test_predictions

    def eval_metrics(self, metrics, only_test=False):
        if not self.model.is_fitted():
            raise RuntimeError('First fitting model!')
        
        test_pool = self.pools.get_test_pool()
        test_metrics = self.model.eval_metrics(test_pool, metrics)
        
        if only_test:
            return test_metrics
        
        train_pool = self.pools.get_train_pool()
        train_metrics = self.model.eval_metrics(train_pool, metrics)
        return train_metrics, test_metrics 
    
    def get_best_score(self):
        if not self.model.is_fitted():
            raise RuntimeError('First fitting model!')
        return self.model.get_best_score()
        
    def _set_parameters(self, params_model):
        for key, value in params_model.items():
            self.set_parameter(key, value)
    
    def _set_common_penalties(self, common_penalty): 
        for feature_name in self.features.get_all_features():
            self.set_feature_penalty(feature_name, common_penalty)
              
    def _set_penalties(self, penalties):
        for feature_name, penalty in penalties.items():
            self.set_feature_penalty(feature_name, penalty)
    
    def _get_text_processing_json(self):
        text_features = self.features.get_text_features()
        if not text_features:
            return {}
        
        text_processing_json = {
            "tokenizers" : [{
                    'tokenizer_id': 'Space',
                    'separator_type': 'ByDelimiter',
                    'token_types': ['Word'],
                    'delimiter' : " "
                }],
            
                "dictionaries" : [{
                    "dictionary_id" : "Trigram",
                    "max_dictionary_size" : "5000",
                    "occurrence_lower_bound" : "100",
                    "gram_order" : "3"
                }, {
                    "dictionary_id" : "BiGram",
                    "max_dictionary_size" : "5000",
                    "occurrence_lower_bound" : "100",
                    "gram_order" : "2"
                }, {
                    "dictionary_id" : "Word",
                    "max_dictionary_size" : "5000",
                    "occurrence_lower_bound" : "100",
                    "gram_order" : "1"
                }, {
                    "dictionary_id" : "Letters",
                    "token_level_type" : "Letter",
                    "max_dictionary_size" : "5000",
                    "occurrence_lower_bound" : "100",
                    "gram_order" : "6"
                }],
                "feature_processing" : {
                    "default" : [{
                        "tokenizers_names" : ["Space"],
                        "dictionaries_names" : ["Word"],
                        "feature_calcers" : ["BoW"]
                    }]
                }
            }
        
        if 'ListOfMedicines' in text_features:
            index_list_of_medicines = text_features.index('ListOfMedicines')
            feature_processing = text_processing_json['feature_processing']
            feature_processing[str(index_list_of_medicines)] = [{
                "tokenizers_names" : ["Space"],
                "dictionaries_names" : ["Letters", "BiGram", "Trigram"],
                "feature_calcers" : ["BoW"]
            }]
        return text_processing_json
    
    def get_feature_importance(self, prettified=True):
        if not self.model.is_fitted():
            raise RuntimeError("The model doesn't fitted!")
        
        return self.model.get_feature_importance(type=EFstrType.FeatureImportance, prettified=prettified)
        
    def get_feature_interactions(self, prettified=True):
        if not self.model.is_fitted():
            raise RuntimeError("The model doesn't fitted!")
        
        return self.model.get_feature_importance(type=EFstrType.Interaction, prettified=prettified)
  
