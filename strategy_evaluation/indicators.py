import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data

def normalization(df, symbol):
    df= df.dropna()
    return df[symbol] / df[symbol][0]

def ema(sd, ed, symbol, plot = False):

    historical_time = dt.timedelta(50)
    new_sd = sd - historical_time

    # Get Data for stock
    price_df = get_data([symbol], pd.date_range(new_sd, ed))
    price_df = price_df[[symbol]]
    price_df = price_df.ffill().bfill()

    #Setting window size
    ema_20 = 20
    ema_50 = 50
    #Compute the EMA using pandas ewm
    ema_20_df = price_df.ewm(span = ema_20, adjust=False).mean()
    ema_50_df = price_df.ewm(span = ema_50, adjust = False).mean()

    ema_20_df = ema_20_df.truncate(before = sd)
    ema_50_df = ema_50_df.truncate(before = sd)

    # # Normalization for graph
    price_df_norm = price_df[symbol] / price_df[symbol][0]
    ema_20_norm = ema_20_df[symbol] / ema_20_df[symbol][0]
    ema_50_norm = ema_50_df[symbol] / ema_50_df[symbol][0]


    if plot:
        plt.figure(figsize=(12,10))
        plt.plot(price_df_norm, label ='Stock Price - JPM', color = 'blue')
        plt.plot(ema_20_norm, label = '20 Day EMA', color = 'Red')
        plt.plot(ema_50_norm, label = '50 Day EMA', color = 'Green')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.title('Exponential Moving Average Indicator')
        plt.savefig("images/ema.png", bbox_inches='tight')
        plt.clf()


    return ema_20_df, ema_50_df

def bollinger_band(sd, ed, symbol):
    # Get Data for stock
    price_df = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    price_df = price_df.ffill().bfill()
    price_norm = normalization(price_df, symbol)

    #compute simple moving average  20 day
    sma = price_norm.rolling(20).mean()

    #compute bollinger bands
    std = price_norm.rolling(20).std()
    upper_band = sma + std * 2
    lower_band = sma - std * 2

    plt.figure(figsize=(15,12))
    # plt.plot(price_norm, label ='Stock Price', color = 'black')
    plt.plot(upper_band, label = 'Upper Band', color = 'red')
    plt.plot(lower_band, label = 'Lower Band', color = 'red')
    plt.plot(sma, label = 'Simple Moving Average', color = 'blue')
    plt.plot(price_norm , label = 'Stock Price - JPM', color = 'grey')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.title('Bollinger Bands Indicator')
    plt.savefig("images/bollinger_bands.png", bbox_inches='tight')
    plt.clf()


def macd(sd, ed, symbol, plot = False):
    historical_time = dt.timedelta(50)
    new_sd = sd - historical_time
    # Get Data for stock
    price_df = get_data([symbol], pd.date_range(new_sd, ed))
    price_df = price_df[[symbol]]
    price_df = price_df.ffill().bfill()
    price_norm = normalization(price_df, symbol)

    ema_26 = price_df.ewm(span = 26, adjust = False).mean()
    ema_12 = price_df.ewm(span = 12, adjust= False).mean()

    macd = ema_12 - ema_26
    signal = macd.ewm(span = 9, adjust= False).mean()
    hist = macd - signal


    macd = macd.truncate(before = sd)
    signal = signal.truncate(before = sd)
    hist = hist.truncate(before = sd)

    df = pd.DataFrame()
    df['MACD'] = macd[symbol]
    df['Signal'] = signal[symbol]
    df['Histogram'] = hist[symbol]

    if plot:
        plt.figure(figsize=(15, 12))
        ax1 = plt.subplot2grid((8,1), (0,0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((8,1), (6,0), rowspan=3, colspan=1)
        ax1.plot(price_norm, label = 'Stock Price - JPM')
        ax1.set_title('Stock Price with MACD')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()

        ax2.plot(df['MACD'], color = 'red', label = 'MACD')
        ax2.plot(df['Signal'], color = 'blue', label = 'Signal')
        ax2.set_title('Moving Average Convergence Divergence Indicator')
        ax2.legend()

        for i in range(len(price_norm)):
            if df['Histogram'][i] < 0:
                ax2.bar(price_norm.index[i], df['Histogram'][i], color = 'red')
            else:
                ax2.bar(price_norm.index[i], df['Histogram'][i], color = 'green')
        # plt.legend()
        plt.savefig("images/macd.png", bbox_inches='tight')
        plt.clf()

    return macd, signal

def sma(sd, ed, symbol):
    price_df = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    price_df = price_df.ffill().bfill()
    price_norm = normalization(price_df, symbol)

    sma = price_norm.rolling(20).mean()

    plt.figure(figsize=(12,10))
    plt.title('Simple Moving Average Indicator')
    plt.plot(price_norm, label = 'Stock Price- JPM', color = 'Blue')
    plt.plot(sma, label = 'Simple Moving Average', color = 'Red')
    plt.legend()
    plt.xlabel('Dates')
    plt.ylabel('Normalized Price')
    plt.savefig("images/sma.png", bbox_inches='tight')
    plt.clf()

def tsi(sd, ed, symbol, plot = False):
    historical_time = dt.timedelta(70)
    new_sd = sd - historical_time

    price_df = get_data([symbol], pd.date_range(new_sd, ed))
    price_df = price_df[[symbol]]
    price_df = price_df.ffill().bfill()
    price_norm = normalization(price_df, symbol)

    difference = price_df - price_df.shift(1)
    smooth = difference.ewm(span=25, adjust=False).mean()
    double_smooth = smooth.ewm(span=13, adjust=False).mean()

    # calculate, smoothing and double smoothing absolute price change
    abs_diff = abs(difference)
    abs_smooth = abs_diff.ewm(span=25, adjust=False).mean()
    abs_double_smooth = abs_smooth.ewm(span=13, adjust=False).mean()

    tsi = (double_smooth/ abs_double_smooth)
    signal = tsi.ewm(span = 12, adjust= False).mean()

    tsi = tsi.truncate(before = sd)
    signal = signal.truncate(before = sd)

    price_norm = price_norm.to_frame()
    price_norm['TSI'] = tsi
    price_norm['Signal'] = signal

    if plot:
        plt.figure(figsize=(15, 10))
        ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
        ax1.plot(price_norm['JPM'], label ='Stock Price - JPM')
        ax1.set_title('Stock Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()

        ax2.plot(price_norm['TSI'], linewidth=2, color='orange', label='TSI ')
        ax2.plot(price_norm['Signal'], linewidth=2, color='red', label='SIGNAL')
        ax2.set_title('True Strength Index')
        ax2.set_xlabel('Date')
        ax2.legend()

        plt.savefig("images/tsi.png", bbox_inches='tight')
        plt.clf()

    return tsi, signal


def run(sd, ed, symbol):
    sma(sd, ed, symbol)
    ema(sd, ed, symbol)
    macd(sd, ed, symbol)
    bollinger_band(sd, ed, symbol)
    tsi(sd, ed, symbol)

def author():
    return 'wjo31'


