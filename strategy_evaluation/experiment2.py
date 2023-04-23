from StrategyLearner import StrategyLearner
from ManualStrategy import get_order_df
from marketsimcode import compute_portvals
import datetime as dt
import matplotlib.pyplot as plt

def exp_2(verbose = False):

    def compute_portval(impact_val, sd, ed, sv, symbol):
        strategy_learner = StrategyLearner(impact= impact_val)
        strategy_learner.add_evidence(symbol, sd, ed, sv)
        strategy_trades = strategy_learner.testPolicy(symbol, sd, ed)
        order_df_strategy = get_order_df(strategy_trades)
        strategy_portval = compute_portvals(order_df_strategy, sv, commission=0.00, impact= impact_val)

        return strategy_portval

    sv = 100000
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    impact_test = [0.001, 0.008, 0.016]
    portvals = {}
    for i in impact_test:
        portvals[i] = compute_portval(impact_val= i, sd=sd, ed=ed, sv=sv, symbol=symbol)

    #comparing statistics
    if verbose:
        compare_stats(portvals[0.001], portvals[0.008], portvals[0.016])

    compare_plots(portvals[0.001], portvals[0.008], portvals[0.016])



def compare_stats(benchmark_data, optimal_data, manual_data):
    """
    The function is from Experiment 1 and therefore parameters have the same name (do not get confused since input will
    be 3 different impact measurements)
    """

    #Statistics

    #Cumulative Return
    cum_ret_bench = benchmark_data[-1] / benchmark_data[0] -1
    cum_ret_opt = optimal_data[-1]/optimal_data[0] -1
    cum_ret_man = manual_data[-1]/manual_data[0] -1


    #Daily Returns
    daily_ret_bench, daily_ret_opt, daily_ret_man = benchmark_data.copy(), optimal_data.copy(), manual_data.copy()

    daily_ret_bench[1:] = (daily_ret_bench[1:] / daily_ret_bench[:-1].values) - 1
    daily_ret_bench = daily_ret_bench[1:]

    daily_ret_opt[1:] = (daily_ret_opt[1:] / daily_ret_opt[:-1].values) - 1
    daily_ret_opt = daily_ret_opt[1:]

    daily_ret_man[1:] = (daily_ret_man[1:] / daily_ret_man[:-1].values) - 1
    daily_ret_man = daily_ret_man[1:]

    #Standard Deviation
    std_bench, std_opt, std_man = round(daily_ret_bench.std(),6) , round(daily_ret_opt.std(),6), round(daily_ret_man.std(),6)

    #Mean Daily Returns
    mu_bench, mu_opt, mu_man = round(daily_ret_bench.mean(),6), round(daily_ret_opt.mean(),6), round(daily_ret_man.mean(),6)

    #Sharpe Ratio
    sr_bench, sr_opt, sr_man = round((mu_bench / std_bench), 6),round((mu_opt / std_opt), 6),round((mu_man / std_man), 6)

    print(f'Cumulative Return of Impact 0.001: {cum_ret_bench}  vs Impact 0.008: {cum_ret_opt} vs Impact 0.016: {cum_ret_man}')
    print(f'Standrd Deviation of Daily Returns for Impact 0.001: {std_bench} vs Impact 0.008: {std_opt}v s Impact 0.016: {std_man}')
    print(f'Mean Daily Returns of Impact 0.001: {mu_bench} vs Impact 0.008: {mu_opt} vs Impact 0.016: {mu_man}')
    print(f'Sharpe Ratio of Impact 0.001: {sr_bench} vs Impact 0.008: {sr_opt} vs Impact 0.016: {sr_man}')


    return

def compare_plots(benchmark_data, optimal_data, manual_data):
    benchmark_norm = benchmark_data/ benchmark_data[0]
    optimal_norm = optimal_data/ optimal_data[0]
    manual_norm = manual_data/manual_data[0]

    plt.figure(figsize=(18 ,10))
    plt.title("Strategy Learner - Response to Impacts")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative Return")
    plt.plot(benchmark_norm, label="Impact 0.001", color="purple")
    plt.plot(optimal_norm, label="Impact 0.008", color="red")
    plt.plot(manual_norm, label ="Impact 0.016", color = 'Blue')
    plt.legend()
    plt.savefig(f"images/experiment2.png", bbox_inches='tight')
    plt.clf()

def author():
    return 'wjo31'

if __name__ == '__main__':
    exp_2()