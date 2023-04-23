from util import get_data, plot_data
import datetime as dt
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import indicators
import warnings

warnings.filterwarnings("ignore")


class ManualStrategy:

    def testPolicy(self, sd, ed, symbol, sv):

        #importing the price dataframe and compute indicator values
        price_df, trades_df = price_df_normalized(sd, ed, symbol)
        indicators_df = compute_indicators(sd, ed, symbol)
        dates = price_df.index

        current_position = 0
        for i in range(len(dates)):
            today = dates[i]
            action = action_condition(price_df.loc[today][symbol] ,indicators_df.loc[today]['EMA20'], indicators_df.loc[today]['EMA50'],
                                      indicators_df.loc[today]['MACD'], indicators_df.loc[today]['MACD Signal']
                                      , indicators_df.loc[today]['TSI'],indicators_df.loc[today]['TSI Signal'] ,current_position)

            trades_df.loc[today][symbol] = action
            # print(f'Current Position {current_position}')
            current_position += action
            # print(f'Currnet Position + Action:{action} = {current_position}')

        return trades_df



def price_df_normalized(sd, ed, symbol):
    df = get_data([symbol], pd.date_range(sd, ed))
    price_df = df[[symbol]]
    price_df = price_df.ffill().bfill()
    norm_df = price_df[symbol]/price_df[symbol][0]
    trade_df = df[['SPY']].rename(columns = {'SPY': symbol})

    return price_df , trade_df

def compute_indicators(sd, ed, symbol):
    ema_20, ema_50 = indicators.ema(sd,ed,symbol)
    macd , macd_signal = indicators.macd(sd, ed, symbol)
    tsi , tsi_signal= indicators.tsi(sd, ed, symbol)

    #Generating Indicators Dataframe (ordered by the date)
    ema_20.rename(columns = {symbol: 'EMA20'}, inplace = True)
    ema_50.rename(columns = {symbol: 'EMA50'}, inplace = True)
    macd.rename(columns = {symbol: 'MACD'}, inplace = True)
    macd_signal.rename(columns = {symbol: 'MACD Signal'}, inplace = True)
    tsi.rename(columns = {symbol: 'TSI'}, inplace = True)
    tsi_signal.rename(columns = {symbol: 'TSI Signal'}, inplace = True)

    dataframes = [ema_20, ema_50 ,macd, macd_signal ,tsi, tsi_signal]
    indicator = pd.concat(dataframes, axis = 'columns')

    return indicator

def action_condition(stock_price, ema20, ema50 ,macd, macdsignal, tsi, tsisginal,position):

    #EMA20 Decisions
    ema_difference = 'Long' if (ema20 - stock_price) > 0 else 'Short' if (ema20 - stock_price) < 0 else 'Stay'
    ema_map = {'Long': 1, 'Short':-2, 'Stay': 0}
    ema_score= ema_map[ema_difference]

    ema50_difference = 'Long' if (ema50 - stock_price) > 0 else 'Short' if (ema50 - stock_price) < 0 else 'Stay'
    ema50_map = {'Long': 1, 'Short':-1, 'Stay': 0}
    ema50_score= ema50_map[ema50_difference]

    #MACD Decisions
    macd_difference = 'Long' if macd < macdsignal and macd > 0 else 'Short' if macd > macdsignal else 'Stay'
    macd_map = {'Long': 2, 'Short':-6, 'Stay': -1}
    macd_score= macd_map[macd_difference]

    #TSI Decisions
    tsi_difference = 'Long' if tsi > tsisginal else 'Short' if tsi < tsisginal else 'Stay'
    tsi_map = {'Long': 1, 'Short':-3, 'Stay': 0}
    tsi_score= tsi_map[tsi_difference]

    total_points = ema_score + ema50_score +  macd_score + tsi_score
    decision = 'Long' if total_points >= 3 else 'Short' if total_points <= -5 else 'Stay'
    decision_map = {'Long': 1000 - position, 'Short': -1000 - position, 'Stay': -position}
    final_action = decision_map[decision]

    return final_action


def get_order_df(df_trades):
    from collections import defaultdict
    d = defaultdict(list)
    symbol = df_trades.columns[0]

    dates = df_trades.index
    for date in dates:
        shares = df_trades.loc[date][symbol]
        if shares != 0:
            if shares > 0:
                d['Symbol'].append(symbol)
                d['Order'].append('BUY')
                d['Date'].append(date)
                d['Shares'].append(abs(shares))
            else:
                d['Symbol'].append(symbol)
                d['Order'].append('SELL')
                d['Date'].append(date)
                d['Shares'].append(abs(shares))
        continue

    df = pd.DataFrame(dict(d))
    return df.set_index('Date')

def benchmark(sd, ed, sv):

    def get_bench_df(df_trades):
        from collections import defaultdict
        d = defaultdict(list)
        symbol = df_trades.columns[0]

        dates = df_trades.index
        for date in dates:
            shares = df_trades.loc[date][symbol]
            if shares != 0:
                if shares > 0:
                    d['Symbol'].append(symbol)
                    d['Order'].append('BUY')
                    d['Date'].append(date)
                    d['Shares'].append(abs(shares))
                else:
                    d['Symbol'].append(symbol)
                    d['Order'].append('SELL')
                    d['Date'].append(date)
                    d['Shares'].append(abs(shares))
            else:
                d['Symbol'].append(symbol)
                d['Order'].append(0)
                d['Date'].append(date)
                d['Shares'].append(0)
            continue

        df = pd.DataFrame(dict(d))
        return df.set_index('Date')

    df = get_data(['SPY'], pd.date_range(sd, ed))
    df = df.rename(columns={'SPY': 'JPM'}).astype({'JPM': 'int32'})
    df[:] = 0
    df.loc[df.index[0]] = 1000
    order_df = get_bench_df(df)
    benchmark_df = compute_portvals(order_df, sv, commission=0.00, impact=0.00)
    return benchmark_df

def compare_stats(benchmark_data, optimal_data):

    #Statistics

    #Cumulative Return
    cum_ret_bench = benchmark_data[-1] / benchmark_data[0] -1
    cum_ret_opt = optimal_data[-1]/optimal_data[0] -1

    #Daily Returns
    daily_ret_bench, daily_ret_opt = benchmark_data.copy(), optimal_data.copy()

    daily_ret_bench[1:] = (daily_ret_bench[1:] / daily_ret_bench[:-1].values) - 1
    daily_ret_bench = daily_ret_bench[1:]

    daily_ret_opt[1:] = (daily_ret_opt[1:] / daily_ret_opt[:-1].values) - 1
    daily_ret_opt = daily_ret_opt[1:]

    #Standard Deviation
    std_bench, std_opt = round(daily_ret_bench.std(),6) , round(daily_ret_opt.std(),6)

    #Mean Daily Returns
    mu_bench, mu_opt = round(daily_ret_bench.mean(),6), round(daily_ret_opt.mean(),6)

    print(f'Cumulative Return of Benchmark: {cum_ret_bench}  vs Manual: {cum_ret_opt}')
    print(f'Standrd Deviation of Daily Returns for Benchmark: {std_bench} vs Manual: {std_opt}')
    print(f'Mean Daily Returns of Benchmark: {mu_bench} vs Manual: {mu_opt}')

def compare_plots(benchmark_data, optimal_data, orders_df, sample_type):
    benchmark_norm = benchmark_data/ benchmark_data[0]
    optimal_norm = optimal_data/ optimal_data[0]

    plt.figure(figsize=(18 ,10))
    plt.title(f"Benchmark vs Manual Returns - {sample_type}")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative Return")
    plt.plot(benchmark_norm, label="Benchmark", color="purple")
    plt.plot(optimal_norm, label="Manual", color="red")

    for i in range(len(orders_df)):
        if orders_df['Order'][i] == 'BUY':
            plt.axvline(orders_df.index[i], color = 'Blue')
        else:
            plt.axvline(orders_df.index[i], color = 'Black' )

    plt.legend()
    plt.savefig(f"images/Manual-{sample_type}.png", bbox_inches='tight')
    plt.clf()

def report(in_sample = True, verbose = False):
    sv = 100000
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1) if in_sample else dt.datetime(2010, 1, 1)
    ed = dt.datetime(2009, 12, 31) if in_sample else dt.datetime(2011,12,31)

    file_name = 'In_Sample' if in_sample else 'Out_Sample'

    #benchmark data
    bench = benchmark(sd, ed, sv)

    #Manual Strategy
    manual = ManualStrategy()
    df_trades = manual.testPolicy(sd=sd, ed=ed, symbol= symbol,sv=sv)
    order_df = get_order_df(df_trades)
    manual_portvals = compute_portvals(order_df, sv, commission=9.95, impact=0.005)

    #Statistics and Plot Comparison
    if verbose:
        compare_stats(bench, manual_portvals)

    compare_plots(bench, manual_portvals, order_df, file_name)


def author():
    return 'wjo31'


if __name__ == "__main__":
    report(in_sample=True, verbose=False)
    report(in_sample=False, verbose=False)

