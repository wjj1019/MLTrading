""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt

  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def author():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    return "wjo31"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def gtid():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    return 903664864  # replace with your GT ID number
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  	  		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    result = False  		  	   		  	  		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		  	  		  		  		    	 		 		   		 		  
        result = True  		  	   		  	  		  		  		    	 		 		   		 		  
    return result  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
# Set Seed
np.random.seed(gtid())


def test_code():
    """
    Method to test your code
    """
    win_prob = 9 / 19  # set appropriately to the probability of a win

    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    def simulation(prob: float, budget=False):
        """
        Performs Betting simulation
        Args:
            cap: Capping value for winnings (when reached cap, stop betting)
            prob: winning probability
        """

        # Cumulative winnings per episode
        episode_winnings = 0
        # Keeping track of winnings for 1000 bets (filled up array with 80s)
        winnings = np.full(shape=(1001), fill_value=80, dtype=np.int64)
        # Counting the number of bets
        bet_count = 0

        # Budget is either 256 (Experiment 2) or Unlimited(Experiment 1)
        budget = 256 if budget == True else np.inf

        # keeping playing if did not reach the target amount
        while episode_winnings < 80:
            won = False
            bet_amount = 1

            while not won:
                # Storing amount to the array
                winnings[bet_count] = episode_winnings
                # if betting count is over 1000 just return the array
                if bet_count >= 1000:
                    return winnings
                # Increasing the bet count for next betting
                bet_count = bet_count + 1
                won = get_spin_result(prob)
                # Keeping Track of cumulative earnings (starts from $0) and Budget (Starts from $256)
                if won == True:
                    episode_winnings = episode_winnings + bet_amount
                    budget = budget + bet_amount
                else:
                    episode_winnings = episode_winnings - bet_amount
                    budget = budget - bet_amount

                    # If current budget is less than the next betting amount (suppose to be 2* previous betting) just bet all of the remainder amount
                    if budget < bet_amount * 2:
                        bet_amount = budget
                    # If there is enough budget for 2 fold increase from previous bet, do it
                    else:
                        bet_amount = bet_amount * 2

                    # if current winnings is at -256
                    if episode_winnings == -256:
                        # when the budget reached -256, fill up the remaining array with -256
                        winnings[bet_count:] = -256
                        # print('Exit at -256')
                        return winnings
        # Filling th remaining columns with cap value (80 or -256)
        return winnings

    def episodes(num_episodes: int, statistics=False, budget=False):
        """
        Get the simulation result and descriptive statistics for n episodes
        Args:
            num_episodes: number of episodes to run the simulation
            cap: Capping value for winning
            statistics: choice of statsitics
        Returns:
            numpy matrix [n_episodes + (mean/median) + stdev, sereis of betting]
            or [n_episodes, series of betting] if no statistics given
        """
        # initialize the size of the matrix [Episode Number , Series of betting]
        episodes_matrix = np.empty(shape=(num_episodes, 1001))
        # Looping through all the number of episodes given
        for episode in range(num_episodes):
            # Retrieve the winning records from simulation function
            sim_result = simulation(prob=win_prob, budget=budget)
            # Setting each row of the matrix with arrays with 1001 (from simulation function)
            episodes_matrix[episode, :] = sim_result

        # Computing standard deviation of generated episodes matrix
        stdev = episodes_matrix.std(axis=0)

        # Vertical stacking of numpy array with column wise mean or median else no stats.
        if statistics == True:
            mean = np.vstack((episodes_matrix, episodes_matrix.mean(axis=0)))
            mean_data = np.vstack((mean, stdev))
            median = np.vstack((episodes_matrix, np.median(episodes_matrix, axis=0)))
            median_data = np.vstack((median, stdev))
            return mean_data, median_data
        return episodes_matrix

    fig_1 = episodes(num_episodes=10,  statistics=False, budget=False)
    fig_2, fig_3 = episodes(num_episodes=1000, statistics=True, budget=False)
    fig_4, fig_5 = episodes(num_episodes=1000 , statistics=True, budget=True)

    def figure1(data):
        fig = plt.figure(figsize=(10, 8))
        plt.xlim((0, 300))
        plt.ylim((-256, 100))
        plt.title('Figure1')
        plt.xlabel('Number of Spins')
        plt.ylabel('Cumulative Winnings')
        plt.plot(data[0, :], label='Episode1')
        plt.plot(data[1, :], label='Episode2')
        plt.plot(data[2, :], label='Episode3')
        plt.plot(data[3, :], label='Episode4')
        plt.plot(data[4, :], label='Episide5')
        plt.plot(data[5, :], label='Episode6')
        plt.plot(data[6, :], label='Episode7')
        plt.plot(data[7, :], label='Episode8')
        plt.plot(data[8, :], label='Episode9')
        plt.plot(data[9, :], label='Episode10')
        plt.legend(loc='lower right')
        plt.show()
        fig.savefig('images/figure_1.png')
        plt.close(fig)
    #Save figure1 to images subdirectory
    figure1(fig_1)

    def create_plot(data, i, legend):
        fig = plt.figure(figsize=(10, 8))
        plt.xlim((0, 300))
        plt.ylim((-256, 100))
        plt.xlabel('Number of Spins')
        plt.ylabel('Cumulative Winnings')
        plt.title('Figure{}'.format(i))
        # Plotting either the mean or median (2nd last row of the matrix)
        plt.plot(data[-2, :], label=legend[0])
        # Upper and lower of the Standard Deviation (Std at last row of the matrix)
        upper = data[-2, :] + data[-1, :]
        lower = data[-2, :] - data[-1, :]
        # plotting upper and lower bound
        plt.plot(upper, label=legend[1])
        plt.plot(lower, label=legend[2])
        plt.legend()
        # Saving files into images subdirectory and close to not make the plot show
        fig.savefig('images/figure_{}.png'.format(i))
        plt.close(fig)

    # Initializing legend names
    legend_mean = ['Mean', 'Std Above Mean', 'Std Below Mean']
    legend_median = ['Median', 'Std Above Median', 'Std Below Median']

    # Looping through  figure 2->5 (because one requires different input)
    for num, data in zip([i for i in range(2, 6)], [fig_2, fig_3, fig_4, fig_5]):
        # Figure 2 and 4 is the mean graph
        if num in [2, 4]:
            legend = legend_mean
        # Figure 3 and 5 is the median graph
        else:
            legend = legend_median
        create_plot(data, num, legend)

  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    test_code()  		  	   		  	  		  		  		    	 		 		   		 		  



















