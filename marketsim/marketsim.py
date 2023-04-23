""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
import os  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
from util import get_data, plot_data


def compute_portvals(
        orders_file="./orders/orders.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # Reading the orderbook
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    # Creating Price dataframe
    symbols = orders_df['Symbol'].unique()
    start_date, end_date = orders_df.index[0], orders_df.index[-1]
    standard_df = get_data(['SPY'], pd.date_range(start_date, end_date), addSPY=True, colname = 'Adj Close')


    price_df = get_data(symbols, pd.date_range(start_date, end_date), addSPY=False, colname='Adj Close')
    price_df = price_df.ffill().bfill()

    price_df = standard_df.join(price_df,how = 'inner')

    # print(len(price_df))

    # Creating trades dataframe
    unique_dates = orders_df.index.unique()
    # trade_df = pd.DataFrame(np.zeros((len(orders_df), len(symbols) )), index = list(orders_df.index.values), columns = symbols )
    trade_df = price_df.copy()
    trade_df['Cash'] = 1

    # Looping over the dates and fill in the trade_df values
    for i in range(len(trade_df)):
        if trade_df.index[i] not in unique_dates:
            trade_df.iloc[i] = 0
        else:
            date = trade_df.index[i]

            orders = orders_df.loc[date]
            prices = price_df.loc[date]
            # print(orders)
            # print(prices)
            trade_df.loc[date] = order_record(orders, prices, commission, impact)

    trade_df.fillna(0, inplace=True)
    # print(price_df.loc[unique_dates])

    # Creating Holding_df
    start_cash = start_val
    holding_df = trade_df.copy()
    holding_df.iloc[0]['Cash'] = holding_df.iloc[0]['Cash'] + start_cash
    # iterate over the dataframe and the current row will be previous row + current row (updating row each time) - accumulation process
    for i in range(1, len(holding_df)):
        holding_df.iloc[i] = holding_df.ix[i] + holding_df.ix[i - 1]

    # print(trade_df)
    # print(holding_df)

    holdings = holding_df[holding_df.columns[:-1]]

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    #
    # return rv
    # return portvals
    # print(holding_df[300:])
    # print(price_df.loc[unique_dates])
    # print(holding_df[0].index)
    # port_val = holdings.apply(lambda x: daily_portfolio(x.index.values, x.columns[:-1], price_df, holding_df, x['Cash'] ) )
    portvals_df = holding_df.copy()

    values = []
    for i in range(len(portvals_df)):
        port_val = daily_portfolio(portvals_df.index[i], holdings, price_df, portvals_df, portvals_df.iloc[i]['Cash'])
        # print(port_val, portvals_df.index[i])
        values.append(port_val)

    portvals_df['PortVals'] = values
    print(portvals_df['PortVals'])
    return portvals_df['PortVals']



def order_record(orders, price_df, transaction, impact):
    """
    Args:
        orders: dataframe or series from order book with certain dates
        price_df: pd.series from price_df (price of each stock on particular day
    """
    from collections import defaultdict
    record = defaultdict(float)
    cash = 0
    # check if more than one orders are made in a single day (will result in pd.dataframe)
    if isinstance(orders, pd.DataFrame):
        for i in range(len(orders)):
            symbol = orders.iloc[i]['Symbol']  # price of Stock
            shares = orders.iloc[i]['Shares']  # Shares of stock
            order = orders.iloc[i]['Order']  # Buy or sell of stock
            if order == 'BUY':
                record[symbol] = record[symbol] + shares
                cash = cash + (-price_df[symbol] * (1 + impact) * shares) - transaction
                record['Cash'] = cash
            elif order == 'SELL':
                record[symbol] = record[symbol] - shares
                cash = cash + (price_df[symbol] * (1 - impact) * shares) - transaction
                record['Cash'] = cash


    # single order on a particular day will result in pd.series
    else:
        symbol = orders['Symbol']
        shares = orders['Shares']
        order = orders['Order']
        if order == 'BUY':
            record[symbol] = record[symbol] + shares
            cash = cash + (-price_df[symbol] * (1 + impact) * shares) - transaction
            record['Cash'] = cash
        elif order == 'SELL':
            record[symbol] = record[symbol] - shares
            cash = cash + (price_df[symbol] * (1 - impact) * shares) - transaction
            record['Cash'] = cash

    return pd.Series(record)

def daily_portfolio(date, symbols, price_data, holdings_data,current_cash):
    stock_prices = 0
    for stock in symbols:
        stock_prices += price_data.loc[date][stock] * holdings_data.loc[date][stock]
    return stock_prices + current_cash

def author():
    return 'wjo31'

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals('./orders/orders-11.csv',start_val=sv, commission=100, impact=0.006)

    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]  # just get the first column
    # else:
    #     "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
    #
    # # Compare portfolio against $SPX
    # print(f"Date Range: {start_date} to {end_date}")
    # print()
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    # print()
    # print(f"Cumulative Return of Fund: {cum_ret}")
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    # print()
    # print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    # print()
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    # print()
    # print(f"Final Portfolio Value: {portvals[-1]}")

  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    test_code()
