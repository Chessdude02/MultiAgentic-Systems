import numpy as np

class EnsembleAgent:
    def average(self, predictions: dict):
        return np.mean(list(predictions.values()), axis=0)
