import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def compute_portvals(
        orders_df,
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
    # orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    # Get the Stock name from the main dataframe
    symbols = orders_df['Symbol'].unique()
    #Identifying the start and end date from the main df
    start_date, end_date = orders_df.index[0], orders_df.index[-1]
    # start_date, end_date = dt.datetime(2008, 1,1) , dt.datetime(2009,12,31)

    #In order to get the date range (Since depending on the stock, some day might not be traded)
    standard_df = get_data(['SPY'], pd.date_range(start_date, end_date), addSPY=True, colname='Adj Close')

    #Getting the prices of each stocks within the orders_df
    price_df = get_data(symbols, pd.date_range(start_date, end_date), addSPY=False, colname='Adj Close')
    price_df = price_df.ffill().bfill()

    #Join the standard_df and price_df to only extract the prices of stock on a specific date (SPY as standrd)
    price_df = standard_df.join(price_df, how='inner')

    # Creating trades dataframe
    unique_dates = orders_df.index.unique()
    # trade_df = pd.DataFrame(np.zeros((len(orders_df), len(symbols) )), index = list(orders_df.index.values), columns = symbols )
    trade_df = price_df.copy()
    trade_df['Cash'] = 1

    # Looping over the dates and fill in the trade_df values
    for i in range(len(trade_df)):
        if trade_df.index[i] not in unique_dates:
            #if the dates not in the dataframe, then set it equal to 0
            trade_df.iloc[i] = 0
        else:
            #Initializing the input to order_record function (Date, BUY/SELL order, Price at Date)
            date = trade_df.index[i]
            orders = orders_df.loc[date]
            prices = price_df.loc[date]
            #Apply the function and redefine the date with calculated amount
            trade_df.loc[date] = order_record(orders, prices, commission, impact)

    trade_df.fillna(0, inplace=True)
    # print(trade_df)

    # Creating Holding_df
    start_cash = start_val
    holding_df = trade_df.copy()
    #Setting the first row: adding the start cash to the first row

    holding_df.iloc[0]['Cash'] = holding_df.iloc[0]['Cash'] + start_cash

    # iterate over the dataframe and the current row will be previous row + current row (updating row each time) - accumulation process
    for i in range(1, len(holding_df)):
        holding_df.iloc[i] = holding_df.ix[i] + holding_df.ix[i - 1]

    holdings = holding_df[holding_df.columns[:-1]]

    #Portfolio values dataframe
    portvals_df = holding_df.copy()

    values = []
    #Calculating the daily portfolio values and add a new column with the value
    for i in range(len(portvals_df)):
        port_val = daily_portfolio(portvals_df.index[i], holdings, price_df, portvals_df, portvals_df.iloc[i]['Cash'])
        # print(port_val, portvals_df.index[i])
        values.append(port_val)

    portvals_df['PortVals'] = values
    # print(portvals_df['PortVals'])

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
            else:
                record[symbol] = 0
                record['Cash'] = 0


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
        else:
            record[symbol] = 0
            record['Cash'] = 0
    # print(f'Stock Price {price_df[symbol]} Order {order} Share {shares} Cash {cash}' )
    return pd.Series(record)


def daily_portfolio(date, symbols, price_data, holdings_data, current_cash):
    stock_prices = 0
    for stock in symbols:
        stock_prices += price_data.loc[date][stock] * holdings_data.loc[date][stock]
    return stock_prices + current_cash

def author():
    return 'wjo31'

