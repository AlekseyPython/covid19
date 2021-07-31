from sklearn.model_selection import train_test_split
from Business.AMachineLearning import AMachineLearning
from .Learning import Learning
from .LearningData import LearningData


class XGBoostLearning(AMachineLearning, Learning):
    def __init__(self, source_data, converted_data):
        AMachineLearning.__init__(self)
        Learning.__init__(self)

        self.learning_data = LearningData(source_data, converted_data, cat_boost=False)
