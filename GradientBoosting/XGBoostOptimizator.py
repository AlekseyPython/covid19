import xgboost as xgb
from Business.AMachineLearning import AOptimizatorGradientBoosting
from .Optimizator import Optimizator
from .LearningData import LearningData


class XGBoostOptimizator(AOptimizatorGradientBoosting, Optimizator):
    def __init__(self, source_data, converted_data):
        AOptimizatorGradientBoosting.__init__(self)
        Optimizator.__init__(self)

        self.multiplier_of_zero_class = 2
        self.learning_data = LearningData(source_data, converted_data, cat_boost=False)
        self.xgb = xgb.XGBClassifier()
        
    def train(self, X_train, y_train, X_test, y_test):
        self.xgb.fit(X_train, y_train)
        preds = self.xgb.predict(X_test)
        needed_metrics=set(['Kappa', 'MCC'])
        return self.count_metrics(preds, y_test, needed_metrics, self.multiplier_of_zero_class)
