from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy, benchmark, get_order_df
from marketsimcode import compute_portvals
import datetime as dt
import matplotlib.pyplot as plt

def exp_1(verbose = False):
    #Requires variables
    sv = 100000
    symbol = 'JPM'
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    #Benchmark_data In Sample
    benchmarkin_portval = benchmark(sd_in, ed_in, sv)
    benchmarkout_portval = benchmark(sd_out,ed_out,sv)

    #Training with In Sample Data
    strategy_learner = StrategyLearner(commission=9.95, impact = 0.005)
    strategy_learner.add_evidence(symbol, sd_in, ed_in, sv)

    #Testing with Insample Data
    insample_strategy = strategy_learner.testPolicy(symbol, sd_in, ed_in)
    order_df_instrategy = get_order_df(insample_strategy)
    instrategy_portval = compute_portvals(order_df_instrategy, sv)

    #Testing with Outsample Data
    outsample_strategy = strategy_learner.testPolicy(symbol, sd_out, ed_out)
    order_df_outstrategy = get_order_df(outsample_strategy)
    outstrategy_portval = compute_portvals(order_df_outstrategy, sv)


    #Manual Learner In Sample
    manual_learner = ManualStrategy()
    inmanual_trades = manual_learner.testPolicy(sd_in, ed_in, symbol, sv)
    inorder_df_manual = get_order_df(inmanual_trades)
    inmanual_portval = compute_portvals(inorder_df_manual, sv)

    #Manual Learner Out Sample
    manual_learner = ManualStrategy()
    outmanual_trades = manual_learner.testPolicy(sd_out, ed_out, symbol, sv)
    outorder_df_manual = get_order_df(outmanual_trades)
    outmanual_portval = compute_portvals(outorder_df_manual, sv)



    if verbose:
        compare_stats(benchmarkin_portval, instrategy_portval, inmanual_portval)
        compare_stats(benchmarkout_portval,outstrategy_portval,outmanual_portval)

    compare_plots(benchmarkin_portval, instrategy_portval, inmanual_portval, 'In_Sample')
    compare_plots(benchmarkout_portval, outstrategy_portval, outmanual_portval, 'Out_Sample')


def compare_plots(benchmark_data, optimal_data, manual_data, sample_type):
    benchmark_norm = benchmark_data/ benchmark_data[0]
    optimal_norm = optimal_data/ optimal_data[0]
    manual_norm = manual_data/manual_data[0]

    plt.figure(figsize=(18 ,10))
    plt.title(f"Benchmark vs Strategy vs Manual {sample_type}")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative Return")
    plt.plot(benchmark_norm, label="Benchmark", color="purple")
    plt.plot(optimal_norm, label="Strategy", color="red")
    plt.plot(manual_norm, label ="Manual", color = 'Blue')
    plt.legend()
    plt.savefig(f"images/experiment1-{sample_type}.png", bbox_inches='tight')
    plt.clf()

def compare_stats(benchmark_data, optimal_data, manual_data):

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

    print(f'Cumulative Return of Benchmark: {cum_ret_bench}  vs Strategy: {cum_ret_opt} vs Manual: {cum_ret_man}')
    print(f'Standrd Deviation of Daily Returns for Benchmark: {std_bench} vs Strategy: {std_opt}v s Manual: {std_man}')
    print(f'Mean Daily Returns of Benchmark: {mu_bench} vs Strategy: {mu_opt} vs Manual: {mu_man}')

def author():
    return 'wjo31'

if __name__ == '__main__':
    exp_1(verbose = False)
