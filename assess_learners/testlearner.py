""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import math  		  	   		  	  		  		  		    	 		 		   		 		  
import sys  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np
import math
import matplotlib.pyplot as plt
from time import perf_counter, sleep
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])


    if sys.argv[1] == 'Data/Istanbul.csv':
        data = data[1:,1:]

    data = data.astype('float')
    np.random.seed(903664864)
    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    import math
    import matplotlib.pyplot as plt


    def experiment1(x_train, y_train, x_test, y_test):

        in_sample = []
        out_sample = []

        for leaf in range(1, 101):
            learner = dt.DTLearner(leaf_size=leaf, verbose=False)
            learner.add_evidence(x_train, y_train)
            # In sample prediction
            is_pred = learner.query(x_train)
            os_pred = learner.query(x_test)

            # RMSE computation In and Out of sample
            is_rmse = math.sqrt(((y_train.astype(float) - is_pred) ** 2).sum() / y_train.shape[0])
            os_rmse = math.sqrt(((y_test.astype(float) - os_pred) ** 2).sum() / y_test.shape[0])

            in_sample.append(is_rmse)
            out_sample.append(os_rmse)

        fig = plt.figure(figsize=(8, 8))
        plt.title('DTLearner Error Curve Respect to Leaf Size')
        plt.xlabel('Number of leaf_size')
        plt.ylabel('RMSE')
        plt.plot(in_sample, label='In Sample')
        plt.plot(out_sample, label='Out of Sample')
        plt.legend(loc='lower right')
        fig.savefig('images/figure_1.png')
        plt.close(fig)

    def experiment2(x_train, y_train, x_test, y_test):

        in_sample = []
        out_sample = []

        for leaf in range(1, 101):
            learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size': leaf}, bags=20, boost=False, verbose=False)
            learner.add_evidence(x_train, y_train)
            # In sample prediction
            is_pred = learner.query(x_train)
            os_pred = learner.query(x_test)

            # RMSE computation In and Out of sample
            is_rmse = math.sqrt(((y_train.astype(float) - is_pred) ** 2).sum() / y_train.shape[0])
            os_rmse = math.sqrt(((y_test.astype(float) - os_pred) ** 2).sum() / y_test.shape[0])

            in_sample.append(is_rmse * 100)
            out_sample.append(os_rmse* 100)

        fig = plt.figure(figsize=(8, 8))
        plt.title('Bagging Error with Respect to Leaf Size ')
        plt.xlabel('Number of leaf_size')
        plt.ylabel('RMSE')
        plt.plot(in_sample, label='In Sample')
        plt.plot(out_sample, label='Out of Sample')
        plt.legend(loc='lower right')
        plt.plot()
        plt.show()
        #
        # fig.savefig('images/figure_2.png')
        # plt.close(fig)

    def mae(actual: np.ndarray, predicted: np.ndarray):
        return np.mean(np.abs((actual - predicted)))


    def experiment3(x_train, y_train, x_test, y_test):

        dt_time = []
        rt_time = []
        dt_error = []
        rt_error = []
        # for bag in range(10, 101, 10):
        #     decision_tree = bl.BagLearner(dt.DTLearner, kwargs={'leaf_size': 5}, bags=bag, boost=False, verbose=False)
        #     random_tree = bl.BagLearner(rt.RTLearner, kwargs={'leaf_size': 5}, bags=bag, boost=False, verbose=False)
        #
        #     start = perf_counter()
        #     decision_tree.add_evidence(train_x, train_y)
        #     end = perf_counter()
        #     dt_time.append((end - start))
        #
        #     start = perf_counter()
        #     random_tree.add_evidence(train_x, train_y)
        #     end = perf_counter()
        #     rt_time.append((end - start))

        for leaf in range(1, 100):
            dtl = dt.DTLearner(leaf_size=leaf)
            dtl.add_evidence(x_train, y_train)
            dt_pred = dtl.query(x_test)

            rtl = rt.RTLearner(leaf_size=leaf)
            rtl.add_evidence(x_train, y_train)
            rt_pred = rtl.query(x_test)

            # dt_mae = np.sum(np.absolute((y_test.astype(float) - dt_pred.astype(float) )))
            # rt_mae = np.sum(np.absolute((y_test.astype(float) - rt_pred.astype(float) )))
            dt_mae = mae(y_test, dt_pred)
            rt_mae = mae(y_test, rt_pred)

            dt_error.append(dt_mae)
            rt_error.append(rt_mae)

        # fig = plt.figure(figsize=(8, 8))
        # plt.title('DT vs RT Learner Training Time ')
        # plt.xlabel('Number of Trees')
        # plt.ylabel('Time (sec)')
        # plt.plot(dt_time, label='DTLearner')
        # plt.plot(rt_time, label='RTLearner')
        # plt.legend(loc='lower right')
        # fig.savefig('images/figure_3.png')
        # plt.close(fig)

        fig2 = plt.figure(figsize=(12, 10))
        plt.title('DT vs RT Learner MAE ')
        plt.xlabel('Number of leaf_size')
        plt.ylabel('MAE')
        plt.plot(dt_error, label='DTLearner')
        plt.plot(rt_error, label='RTLearner')
        plt.legend(loc='lower right')
        plt.show()

        # fig2.savefig('images/figure_4.png')
        # plt.close(fig)


    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':5},bags=20,boost= False, verbose=True)
    # learner = dt.DTLearner(leaf_size=50, verbose=False)
    # learner = rt.RTLearner(leaf_size=1, verbose = False)
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions

    rmse = math.sqrt(((train_y.astype(float) - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y.astype(float))
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    print(type(pred_y))
    rmse = math.sqrt(((test_y.astype(float) - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y.astype(float))
    print(f"corr: {c[0,1]}")

    # experiment1(train_x, train_y, test_x, test_y)
    # experiment2(train_x, train_y, test_x, test_y)
    experiment2(train_x, train_y, test_x, test_y)
