""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import warnings  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model = None

    def author(self):
        return "wjo31"

    # quality of split measurement
    def best_feature(self, data_x, data_y):
        # Pearson correlation as
        correlations = {}

        for col in range(data_x.shape[1]):
            # np.corrcoef runtime error (denominator being 0)
            denominator = np.sqrt(np.cov(data_x[:, col], data_x[:, col]) * np.cov(data_y, data_y))
            # denominator = np.std(data_x[:,col])

            if denominator.all() <= 0:
                correlations[col] = 0
                # Correlation computation
            else:
                correlations[col] = np.corrcoef(data_x[:, col], data_y).flatten()[1]

        # correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1])}
        # chosen_feature = list(correlations.keys())[-1]
        feature_idx = 0
        corr = 0
        for feature, corre in correlations.items():
            if corre > corr:
                corr = corre
                feature_idx = feature

        return feature_idx

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
        feature_idx = self.best_feature(data_x, data_y)

        # Median as split value
        split_value = np.median(data_x[:, feature_idx])

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
            print(f'Executing Decision Tree Regressor')
            print(f'Leaf Size {self.leaf_size}  produced tree of {self.model.shape}')
            # print(self.model)

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
            feat = point[int(self.model[node][0])]
            # feat = feature.astype(float)
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
            # print(f'Index {i}')
        return np.array(pred)

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
