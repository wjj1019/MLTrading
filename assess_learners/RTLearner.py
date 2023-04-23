import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model = None

    def author(self):
        return "wjo31"


    def build_tree(self, data_x, data_y):
        """
        Args:
            data: dataframe object
        Returns:
            decision tree
        """
        data_x = data_x.astype(float)
        data_y = data_y.astype(float)

        # Recursion Termination Criterion

        # When the leaf has the given leaf size, split_value will be their mean
        if data_x.shape[0] <= self.leaf_size:
            return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

        # When all the values of the y is identical, return that value as split value
        if len(np.unique(data_y)) == 1:
            return np.array([np.nan, data_y[0], np.nan, np.nan])

        # Feature selection
        feature_idx = np.random.choice(data_x.shape[1], size = 1)[0]
        random1, random2 = np.random.choice(data_x.shape[0], size = 2)
        split_value = (data_x[random1, feature_idx] + data_x[random2, feature_idx])/2

        left_condition = data_x[:, feature_idx] <= split_value
        right_condition = data_x[:, feature_idx] > split_value

        # If after split has the same length of data --> infinite recursion case
        if (len(data_x) == len(data_x[left_condition])) or (len(data_x) == len(data_x[right_condition])):
            return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

            # Recursion of left and right tree building
        left_tree = self.build_tree(data_x[left_condition], data_y[left_condition])
        right_tree = self.build_tree(data_x[right_condition], data_y[right_condition])

        movement = 2 if left_tree.ndim == 1 else left_tree.shape[0] + 1

        root = np.array([feature_idx, split_value, 1, movement])

        # print('Left Tree Sample:{}  Right Tree Sample:{}'.format(left_tree.shape[0], right_tree.shape[0]))
        # print('Split Feature Index:{}  with Split Value:{}'.format(feature_idx, split_value))
        # print(left_tree.ndim)
        return np.row_stack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        # training the model
        self.model = self.build_tree(data_x, data_y)

        if self.verbose:
            print(f'Executing Random Tree Regressor')
            print(f'Leaf Size {self.leaf_size}  produced tree of {self.model.shape}')
            print(self.model)

    def prediction(self, point):
        """
        point: np array (1D)
        """
        # point = point.astype(float)
        node = 0
        # loop until it reaches the leaf
        # print('Length {}'.format(len(self.model) ) )
        while ~np.isnan(self.model[node][0]):
            # selected feature from the data point and corresponding split value
            feature = point[int(self.model[node][0])]
            feat = feature.astype(float)
            split = self.model[node][1]
            # print('feature index {} and split value {}'.format(self.model[node][0], split))
            if feat <= split:
                node = node + 1
                # print('Point goes to left node {}'.format(node))
            elif feat > split:
                node = node + int(self.model[node][-1])
                # print('Point goes to right node {} , Adding {}'.format(node, self.model[node][-1]))
            # print(node)
        return self.model[int(node)][1]

    def query(self, points):
        pred = []
        for i in range(len(points)):
            result = self.prediction(points[i])
            pred.append(result)
            # print("index{}".format(i))
        return np.array(pred)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")





