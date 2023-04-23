from util import get_data, plot_data
import datetime as dt
from datetime import timedelta
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt


class TheoreticallyOptimalStrategy:

    def testPolicy(self, symbol, sd, ed, sv):
        #Because we would want to see one day after the actual given end date
        end_date = ed + timedelta(days = 4)

        #Process of getting JPM price dataframe (Same approach as marketsim from Project 5)
        price_df = get_data([symbol], pd.date_range(sd, end_date), addSPY=False, colname='Adj Close')
        price_df = price_df.ffill().bfill()
        standard_df = get_data(['SPY'], pd.date_range(sd, end_date), addSPY=True, colname='Adj Close')
        price_df = standard_df.join(price_df, how='inner')
        price_df.drop(columns=['SPY'], inplace = True)
        #Get the index
        dates = price_df.index

        # print(price_df[:10])
        share_status = []
        current_share = 0
        for i in range(len(dates) - 1):
            if price_df.loc[dates[i + 1]][symbol] < price_df.loc[dates[i]][symbol]:
                position = -1000 - current_share
                # print('Sell')
            elif price_df.loc[dates[i + 1]][symbol] > price_df.loc[dates[i]][symbol]:
                position = 1000 - current_share
                # print('Buy')
            share_status.append(position)
            # print(f'{dates[i]} Action = +/- 1000 - {current_share} = {position}')
            current_share = current_share + position
            # print(f'Updates Current share {current_share}')

        price_df = price_df.ix[:-1]
        price_df['JPM'] = share_status

        return price_df

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
    df = get_data(['SPY'], pd.date_range(sd, ed))
    df = df.rename(columns={'SPY': 'JPM'}).astype({'JPM': 'int32'})
    df[:] = 0
    df.loc[df.index[0]] = 1000
    order_df = get_order_df(df)

    benchmark_df = compute_portvals(order_df, sv, commission=0.00, impact=0.00)
    return benchmark_df

def result():
    #Initialize parameters
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)


    #Get Benchmark Data
    benchmark_data = benchmark(sd, ed, sv)

    #Get Theoretical Data
    optimal = TheoreticallyOptimalStrategy()
    optimal_df = optimal.testPolicy('JPM', sd, ed, sv)
    order_df = get_order_df(optimal_df)
    optimal_data = compute_portvals(order_df, sv, 0.00, 0.00)

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

    print(f'Cumulative Return of Benchmark: {cum_ret_bench}  vs Theoretical: {cum_ret_opt}')
    print(f'Standrd Deviation of Daily Returns for Benchmark: {std_bench} vs Theoretical: {std_opt}')
    print(f'Mean Daily Returns of Benchmark: {mu_bench} vs Theoretical: {mu_opt}')

    # optimal_data = optimal_data.to_frame()
    # benchmark_data = benchmark_data.to_frame()
    #Normalization for graph
    benchmark_norm = benchmark_data/ benchmark_data[0]
    optimal_norm = optimal_data/ optimal_data[0]

    plt.figure(figsize=(12 ,10))
    plt.title("Benchmark vs Theoretical Returns")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative Return")
    plt.plot(benchmark_norm, label="benchmark", color="purple")
    plt.plot(optimal_norm, label="theoritical", color="red")
    plt.legend()
    plt.savefig("images/comparison.png", bbox_inches='tight')
    plt.clf()


def author():
    return 'wjo31'



if __name__ == "__main__":
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = ['JPM']

    optimal = TheoreticallyOptimalStrategy()
    optimal_df = optimal.testPolicy('JPM', sd, ed, sv)
    order_df = get_order_df(optimal_df)
    optimal_data = compute_portvals(order_df, sv, 0.00, 0.00)

    print(optimal_df)
    # print(order_df.head())
    # print(optimal_data.head())
    #

