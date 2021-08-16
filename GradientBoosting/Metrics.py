import numpy as np
from numba import njit, prange


class LoglossObjective:
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
            weights = np.array(weights)
            
        return self.calc_numba(np.array(approxes), np.array(targets), weights)
    
    @staticmethod
    @njit(fastmath=True)
    def calc_numba(approxes, targets, weights):
        result = []
        for index in prange(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
        
            der1 = targets[index] - p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]
                
            result.append((der1, der2))
        return result
        
    
class LoglossMetric:
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False
    
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        if weight is not None:
            assert len(weight) == len(target)
            weight = np.array(weight)
            
        return self.evaluate_numba(np.array(approxes[0]), np.array(target), weight)
    
    @staticmethod
    @njit(fastmath=True)
    def evaluate_numba(approxes, target, weight):
        error_sum = 0.0
        weight_sum = 0.0
        for i in prange(len(target)):
            e = np.exp(approxes[i])
            p = e / (1 + e)
        
            w = 1.0 if weight is None else weight[i]
                
            weight_sum += w
            error_sum += -w * (target[i] * np.log(p) + (1 - target[i]) * np.log(1 - p))
        return error_sum, weight_sum
    
    
class RmseObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
            weights = np.array(weights)
        return self.calc_numba(np.array(approxes), np.array(targets), weights)
        
    @staticmethod
    @njit(fastmath=True)
    def calc_numba(approxes, targets, weights):
        result = []
        for index in prange(len(targets)):
            der1 = targets[index] - approxes[index]
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result
    
    
class RmseMetric:
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        if weight is not None:
            assert len(weight) == len(target)
            weight = np.array(weight)
            
        return self.evaluate_numba(np.array(approxes[0]), np.array(target), weight)
    
    @staticmethod
    @njit(fastmath=True)
    def evaluate_numba(approxes, target, weight):
        error_sum = 0.0
        weight_sum = 0.0
        for i in prange(len(target)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approxes[i] - target[i])**2)

        return error_sum, weight_sum
    
    
class MaeObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
            weights = np.array(weights)
        return self.calc_numba(np.array(approxes), np.array(targets), weights)
        
    @staticmethod
    @njit(fastmath=True)
    def calc_numba(approxes, targets, weights):
        result = []
        for index in prange(len(targets)):
            approx = approxes[index]
            if targets[index]:
                if approx < 1.:
                    der1 = -1.
                    der2 = 0.#5/(1.-approx)
                elif approx > 1.:
                    der1 = 1.
                    der2 = 0.#5/(approx-1.)
                else:
                    der1 = 0.
                    der2 = 0.
            else:
                if approx > 0.:
                    der1 = 1.
                    der2 = 0.#5/approx
                elif approx < 0.:
                    der1 = -1.
                    der2 = 0.#-5/approx
                else:
                    der1 = 0.
                    der2 = 0.
                    
            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result
    
    
class MaeMetric:
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        if weight is not None:
            assert len(weight) == len(target)
            weight = np.array(weight)
            
        return self.evaluate_numba(np.array(approxes[0]), np.array(target), weight)
    
    @staticmethod
    @njit(fastmath=True)
    def evaluate_numba(approxes, target, weight):
        error_sum = 0.0
        weight_sum = 0.0
        for i in prange(len(target)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * np.abs(approxes[i] - target[i])

        return error_sum, weight_sum
    
    

class AccuracyMetric:
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        if weight is not None:
            assert len(weight) == len(target)
            weight = np.array(weight)
            
        best_class = np.argmax(approxes, axis=0)
        return self.evaluate_numba(np.array(best_class), np.array(target), weight)
        
    @staticmethod
    @njit(fastmath=True)
    def evaluate_numba(best_class, target, weight):
        error_sum = 0
        weight_sum = 0 
        for i in prange(len(target)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * (best_class[i] == target[i])

        return error_sum, weight_sum
    
    