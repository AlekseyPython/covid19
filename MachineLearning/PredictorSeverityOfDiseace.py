import os
from enum import Enum
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

from catboost.utils import get_roc_curve, get_fpr_curve, get_fnr_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics, utils
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from catboost.text_processing import Dictionary

from Deco import performance_meter



class _Stratagy(Enum):
    search_penalties = 'Search penalties'
    search_feature_importance = 'Search feature importance'
    search_optimum_parameters = 'Search optimum parameters'
    fit_model = 'Fitting'
    cross_validation = 'Cross validation'
    
            
class Tuner:
    def __init__(self, data):
        self.data = data
        self.calculation_strategy = _Stratagy.search_feature_importance
        
    @performance_meter    
    def fit(self):
        values_severity_of_disease = self.data.get_all_targets[2:]
        for severity_of_disease in values_severity_of_disease:
            X = self.data.get_X(severity_of_disease)
            Y = self.data.get_Y(severity_of_disease)
            
            model = Model(severity_of_disease)
            if self.calculation_strategy == _Stratagy.search_penalties:
                self._fit_penalties(X, Y, model)
                
            elif self.calculation_strategy == _Stratagy.fit_model:
                self._fit_model(X, Y, model, title_charts=severity_of_disease)
                
            elif self.calculation_strategy == _Stratagy.search_feature_importance:
                self._count_feature_importance(X, Y, model)
                
            elif self.calculation_strategy == _Stratagy.cross_validation:
                self._count_cv(X, Y, model)
                
            elif self.calculation_strategy == _Stratagy.search_optimum_parameters:
                self._grid_search(X, Y, model)
                
            else:
                raise RuntimeError('Choose valid calculation strategy!')
    
    def _fit_penalties(self, X, Y, model): pass
        
  
    def _fit_model(self, X, Y, model, silently=False, title_charts=''):
        cat_features = self._find_cat_features(X)
        text_features = self._get_text_features()
            
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42, shuffle=True, stratify=Y)
        train_pool = Pool(X_train, Y_train, cat_features, text_features)
        test_pool = Pool(X_test, Y_test, cat_features, text_features)

        model.fit(train_pool, test_pool, silently)
        
        if not silently:
            print('TRAIN DATA:')
            train_predictions = model.get_predictions(train_pool)
            self.count_metrics(train_predictions, Y_train, silently=False)
            print('\nTEST DATA:')
        
        test_predictions = model.get_predictions(test_pool)  
        metrics = self.count_metrics(test_predictions, Y_test, silently)
        
#         roc_curve = model.get_roc_curve()
#         self._plot_roc_auc(title_charts, roc_curve)
#         self._plot_fpr_fnr_curve(title_charts, roc_curve)
#         self._plot_calibration_curve(title_charts, test_predictions, Y_test)
#         self._plot_fpr_fnr_dencity(title_charts, test_predictions, Y_test)
        
        return metrics
        
    
    
    def _count_feature_importance(self, severity_of_disease, X, Y):
        feature_importance = {}
        metrics_full = self._fit_model(severity_of_disease, X, Y, silently=True)
        for feature_name in X.columns:
            feature = X[feature_name].copy()
            X = X.drop(feature_name, axis=1)
            
            metrics_drop = self._fit_model(severity_of_disease, X, Y, silently=True)
                
            importance = 100*(metrics_full['Kappa'] - metrics_drop['Kappa'])
#             if importance > 0:
#                 importances = []
#                 feature_shuffled = utils.shuffle(feature, random_state=42)
#                 for iteration in range(3):
#                     feature_shuffled = utils.shuffle(feature_shuffled, random_state=iteration)
#                     feature_shuffled = feature_shuffled.reset_index(drop=True)
#                     X[feature_name] = feature_shuffled
#                     
#                     metrics_shuffled = self._fit_model(severity_of_disease, X, Y, silently=True)
#                     
#                     random_importance = 100*(metrics_shuffled['Kappa'] - metrics_drop['Kappa'])
#                     importances.append(random_importance)
#                     if random_importance >= importance:
#                         break
#                 
#                 importances = np.array(importances)
#                 maximum = importances.max()
#                 upper_bound = round(importances.mean() + 2*importances.std(), 2)  
#                 random_importance = max(maximum, upper_bound)
#             else:
#                 random_importance = 0.0
                
            random_importance = 0.0
            feature_importance[feature_name] = (importance, random_importance)

            #restore the original values
            X[feature_name] = feature
                
        #sort by difference
        feature_importance_sorted = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1][0]-item[1][1], reverse=True)}   
        for feature_name, importances in feature_importance_sorted.items():
            importance, random_importance = importances
            print('Feature = ' + feature_name + '\t importance = ' + str(importance))# + '\t random_importance = ' + str(random_importance))
                        
    def _count_cv(self, X, Y, model):
        cat_features = self._find_cat_features(X)
        text_features = self._get_text_features()
        pool = Pool(X, Y, cat_features, text_features)
        cv_results = model.cross_validation(pool)
        print(tabulate(cv_results, headers='keys', tablefmt='psql'))
    
    def _grid_search(self, X, Y, model):
        cat_features = self._find_cat_features(X)
        text_features = self._get_text_features()
        pool = Pool(X, Y, cat_features, text_features)
        result = model.grid_search(pool)
        print('Best parameters: {}\n'.format(result['params']))
         
        print('Results:')
        cv_results = pd.DataFrame(result['cv_results'])
        self._print_values_metric(cv_results, 'Logloss')
        self._print_values_metric(cv_results, 'AUC', use_min=False)
        self._print_values_metric(cv_results, 'CrossEntropy')
        #print(tabulate(cv_results, headers='keys', tablefmt='psql'))
        
    def _print_values_metric(self, results, metic_name, use_min=True, gap=50):
        print(metic_name + ': ')
        
        values_metric = self._get_values_metric(results, metic_name, gap)
        print(values_metric)
        
        if use_min:
            min_metric = round(results['test-' + metic_name + '-mean'].min(), 4)
            print('Min: ' + str(min_metric))
        else:
            max_metric = round(results['test-' + metic_name + '-mean'].max(), 4)
            print('Max: ' + str(max_metric))
        print('')
            
    def _get_values_metric(self, results, metic_name, gap):
        metric = []
        for index in range(0, len(results), gap):
            value = round(results.loc[index, 'test-'+metic_name+'-mean'], 4)
            metric.append(value)
        return metric
        
    def _plot_roc_auc(self, title_chart, roc_curve):
        (fpr, tpr, _) = roc_curve
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)
        
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('severity_of_disease = ' + title_chart, fontsize=16)
        plt.legend(loc="lower right", fontsize=16)
        plt.show()
        
    def _plot_fpr_fnr_curve(self, title_chart, roc_curve):
        (thresholds, fpr) = get_fpr_curve(curve=roc_curve)
        (_, fnr) = get_fnr_curve(curve=roc_curve)
        
        sns.set_style('darkgrid')
        plt.figure()
        lw = 2
        
        plt.plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
        plt.plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Threshold', fontsize=16)
        plt.ylabel('Error Rate', fontsize=16)
        plt.title('severity_of_disease = ' + title_chart, fontsize=16)
        plt.legend(loc="lower left", fontsize=16)
        plt.show()
    
    def _plot_calibration_curve(self, title_chart, predictions, targets):
        df = pd.DataFrame({'predictions':predictions, 'targets':targets})
        df['rounded_predictions'] = np.round(df['predictions'], 1)
        
        grouped = df['targets'].groupby(df['rounded_predictions'])
        means = grouped.mean()
        sizes = grouped.size()
       
        sns.set_style('darkgrid') 
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('severity_of_disease = ' + title_chart, fontsize=16)
        means.plot(kind='line',ax=ax[0], xlabel='predictions', ylabel='real_values')
        sizes.plot(kind='line',ax=ax[1], xlabel='predictions', ylabel='quantity')
        plt.show()  
        
    def _plot_fpr_fnr_dencity(self, title_chart, predictions, targets):
        df = pd.DataFrame({'predictions':predictions, 'targets':targets})
        df['rounded_predictions'] = np.round(df['predictions'], 2) 
        
        indexes = []
        real_positives = []
        real_negatives = []
        
        step=0.05
        begining=0
        while begining<1:
            end = begining + step
            
            real_positive = df[(df.rounded_predictions>=begining) & (df.rounded_predictions<end) & (df.targets==True)].shape[0]
            real_negative = df[(df.rounded_predictions>=begining) & (df.rounded_predictions<end) & (df.targets==False)].shape[0]
            all_real = real_positive + real_negative
            
            indexes.append(begining)
            if all > 0:
                real_positives.append(real_positive/all_real)
                real_negatives.append(real_negative/all_real)
            else:
                real_positives.append(0)
                real_negatives.append(0)
            begining = end
                
        df = pd.DataFrame({'real_positives':real_positives, 'real_negatives':real_negatives}, index=indexes)
        
        sns.set_style('darkgrid') 
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('severity_of_disease = ' + title_chart, fontsize=16)
        df.plot(kind='line',ax=ax, xlabel='predictions', ylabel='real density')
        plt.show()  
        
           

    