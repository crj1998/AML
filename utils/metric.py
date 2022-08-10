
import numpy as np

class LossMetric:
    def __init__(self):
        self.count = 0
        self.sum = 0
    
    def clear(self):
        self.count = 0
        self.sum = 0

    def update(self, val, cnt):
        self.sum += val * cnt
        self.count += cnt

    def item(self):
        return self.sum/max(1, self.count)
    
    def __str__(self):
        return f"{self.item():6.4f}"
    
    def __repr__(self):
        return self.__str__()

class AccuracyMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.Preds = []
        self.Targets = []

    def clear(self):
        self.Preds = []
        self.Targets = []

    def update(self, preds, targets):
        self.Preds.append(preds)
        self.Targets.append(targets)

    def numpy(self):
        return np.hstack(self.Preds), np.hstack(self.Targets)
    
    def item(self):
        return self.accuracy()

    def accuracy(self):
        preds = np.hstack(self.Preds)
        targets = np.hstack(self.Targets)

        return (targets == preds).sum()/targets.shape[0]
    
    def confusion_matrix(self):
        preds = np.hstack(self.Preds)
        targets = np.hstack(self.Targets)

        cm = np.zeros((self.num_classes, self.num_classes))

        for i, j in zip(preds, targets):
            cm[i, j] += 1
        
        return cm
    
    def __str__(self):
        return f"{self.accuracy():5.2%}"
    
    def __repr__(self):
        return self.__str__()