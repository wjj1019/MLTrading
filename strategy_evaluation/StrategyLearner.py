""""""
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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

import datetime as dt
import random

import pandas as pd
from util import get_data
import random
import QLearner as ql
import indicators
from itertools import product


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        random.seed(903664864)
        self.learner = ql.QLearner(num_states= 19683, num_actions=3, alpha=0.2, gamma=0.9, rar=0.9, radr=0.99, dyna = 300, verbose=False)

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # add your code to do learning here

        #Discretizing the indicators (the size of the discrete will be 9 for each indicators)
        look_up, mapping = discretizing(sd, ed, symbol, discrete= 9)

        #Generating Price and trades dataframe
        price_df, trades_df = price_df_generate(sd, ed, symbol)
        trades_df[:] = 0
        dates = price_df.index

        #Set up the variables
        current_position, previous_position = 0, 0
        current_cash, previous_cash = sv, sv

        #Looping through the dates - (Waiting until convergence not use because the code still works efficiently)
        for i in range(1, len(dates)):
            today = dates[i]

            #Get the new state and compute the reward
            s_prime = current_state(current_position, today, look_up, mapping)
            reward = current_position * price_df.loc[today][symbol] + current_cash - previous_position * price_df.loc[today][symbol] - previous_cash

            #New action will be determined by the query function
            new_action = self.learner.query(s_prime, reward)

            #returns the trade action to take when the new action is given by the Q-table
            trade_action = identify_trade(new_action, current_holding= current_position)

            if self.verbose:
                print(f'Date of {today}')
                print(f'Computed Reward from {dates[i -1]}: {reward}')
                print(f'Current Holdings of {current_position}, Current Cash of {current_cash}')
                print(f'Previous Holdings of {previous_position} Previous Cash of {previous_cash}')
                print(f'Stock {symbol} have price of {price_df.loc[today][symbol]}')
                print(f'Action for today is {new_action} and Will perform Trade of {trade_action}')


            previous_position = current_position
            current_position += trade_action
            trades_df.loc[today][symbol] = trade_action

            impact = self.impact if trade_action > 0 else -self.impact

            previous_cash = current_cash
            current_cash += (-price_df.loc[today][symbol] * (1 + impact) * trade_action) -self.commission

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        look_up, mapping = discretizing(sd, ed, symbol, discrete= 9)
        price_df, trades_df = price_df_generate(sd, ed, symbol)
        trades_df[:] = 0
        dates = price_df.index

        current_position = 0

        for i in range(1, len(dates)):
            today = dates[i]

            #Get New state (Note* there is no reward given to Testing because we don't want to train the Q Table)
            s_prime = current_state(current_position, today, look_up, mapping)

            #Instead of querying, use querystate to get the action directly from the trained Q-table
            new_action = self.learner.querysetstate(s_prime)
            trade_action = identify_trade(new_action, current_holding= current_position)

            if self.verbose:
                print(f'Today Date: {today}')
                print(f'New State of {s_prime} will result action of {new_action}')
                print(f'Resulting Trading Strategy of {trade_action}')

            current_position += trade_action
            trades_df.loc[today][symbol] = trade_action

        return trades_df

def identify_trade(action, current_holding):
    #the actions from the Q-table will be 0,1,2 (Short Stay Long)
    trade_action = {0: -1000 - current_holding, 1: -current_holding, 2: 1000 - current_holding}
    return trade_action[action]

def price_df_generate(sd, ed, symbol):
    df = get_data([symbol], pd.date_range(sd, ed))
    price_df = df[[symbol]]
    price_df = price_df.ffill().bfill()
    trade_df = df[['SPY']].rename(columns = {'SPY': symbol})
    return price_df, trade_df


def discretizing(sd, ed, symbol, discrete):
    #Computing all the indicators (EMA20, EMA50, MACD, TSI)
    ema_20, ema_50 = indicators.ema(sd, ed, symbol)
    macd , macd_signal = indicators.macd(sd, ed, symbol)
    tsi, _ = indicators.tsi(sd,ed,symbol)

    #Discretization using qcut function with 5 integers as label
    ema_20 = pd.qcut(ema_20[symbol], discrete, labels = range(discrete)).to_frame()
    ema_20.rename(columns = {symbol: 'EMA20'}, inplace = True)

    ema_50 = pd.qcut(ema_50[symbol], discrete, labels = range(discrete)).to_frame()
    ema_50.rename(columns = {symbol: 'EMA50'}, inplace = True)

    macd = pd.qcut(macd[symbol], discrete, labels = range(discrete)).to_frame()
    macd.rename(columns = {symbol: 'MACD'}, inplace = True)

    tsi = pd.qcut(tsi[symbol], discrete, labels = range(discrete)).to_frame()
    tsi.rename(columns = {symbol: 'TSI'}, inplace = True)

    dataframes = [ema_20, ema_50, macd, tsi]
    lu_table = pd.concat(dataframes, axis = 'columns')
    # lu_table['State'] = lu_table['EMA20'].astype(str) + lu_table['EMA50'].astype(str) + lu_table['MACD'].astype(str) + lu_table['TSI'].astype(str)

    #Because Q-Table has row of all the states, we want to get all different combinations arising from Indicators + Holdings (3 states for holdings)
    a, b ,c, d = [i for i in range(discrete)], [i for i in range(discrete)], [i for i in range(discrete)],[i for i in range(discrete)]
    e = [i for i in range(3)]

    #The product function will return tuples of all different combinations of 5 x 5 x 5 x 5 x 3 (EMA20, EMA50, MACD, TSI, Holdings)
    comb = list(product(a, b, c, d, e))

    #The dictionary will contain 1875 different combinations but have 0-1875 as their value (this is used to map Q-Table)
    mapping = {comb[i] : i for i in range(len(comb))}
    # print(len(comb))
    return lu_table, mapping

def current_state(position, today ,lu_table, mapping):
    #Checking for current holdings and returns the categorical values {0: -1000 Shares, 1: 0 Shares, 2: 1000 Shares}
    holding_map = {1000: 2, 0: 1, -1000: 2}
    holding = holding_map[position]

    #Since the lookup table from the discretization have all the state combinations for each day retreive their values
    ema20 = lu_table['EMA20'].loc[today]
    ema50 = lu_table['EMA50'].loc[today]
    macd = lu_table['MACD'].loc[today]
    tsi = lu_table['TSI'].loc[today]

    #Setting the tuple in order to call from the dictionary
    state_tuple = (ema20, ema50, macd, tsi, holding)

    #Call the Index of current state combinations
    mapped = mapping[state_tuple]

    return mapped

def author():
    return 'wjo31'


if __name__ == "__main__":
    print("One does not simply think up a strategy")
