import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs: dict, bags: int, boost, verbose=False):
        self.boost = boost
        self.verbose = verbose
        self.learner = learner
        self.learners = None
        self.bags = bags
        self.kwargs = kwargs

    def author(self):
        return "wjo31"

    def add_evidence(self, data_x, data_y):
        self.learners = []
        for _ in range(self.bags):
            self.kwargs = dict(self.kwargs, verbose=self.verbose)
            # initialize the single model
            model = self.learner(**self.kwargs)
            # Random sampling with replacement (size = given data rows)
            sampling = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            x = data_x[sampling]
            y = data_y[sampling]
            # train the model with samplered data
            model.add_evidence(x, y)
            # store each trained model
            self.learners.append(model)

    def query(self, points):
        result = []
        for learner in self.learners:
            result.append(learner.query(points))
        return sum(result) / len(result)

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
