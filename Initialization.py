from ControllerDataSource.IControllerDataSource import IControllerDataSourse
icontroller_data_sourse = IControllerDataSourse()

from DataSource.IDataSourse import IDataSourse
idata_source = IDataSourse()

from Presentation.IPresentation import IPresentation
ipresentation = IPresentation()

from Business.IBusiness import IBusiness
ibusiness = IBusiness()

from GradientBoosting.FabricGradientBoosting import FabricGradientBoosting
fabric_gradient_boosting = FabricGradientBoosting()

from NeuralNetwork.FabricNeuralNetwork import FabricNeuralNetwork
fabric_neural_network = FabricNeuralNetwork()