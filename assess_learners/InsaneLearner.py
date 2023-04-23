import numpy as np
import BagLearner as bl
import LinRegLearner as lr
class InsaneLearner(object):
    def __init__(self, verbose):
        self.verbose = verbose
        self.learners = [bl.BagLearner(learner=lr.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in
                         range(20)]
    def author(self):
        return "wjo31"
    def add_evidence(self, data_x, data_y):
        for model in self.learners:
            model.add_evidence(data_x, data_y)
    def query(self, points):
        result = []
        for model in self.learners:
            result.append(model.query(points))
        return sum(result) / len(result)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
