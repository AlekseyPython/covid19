from Business.AMachineLearning import AFabricNeuralNetwork
from .PytorchOptimizator import PytorchOptimizator

class FabricNeuralNetwork(AFabricNeuralNetwork):
    def __init__(self):
        AFabricNeuralNetwork.__init__(self)
    
    def create_optimizator(self, source_data, converted_data):
        return PytorchOptimizator(source_data, converted_data)
        