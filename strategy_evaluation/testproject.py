import experiment1
import experiment2
import ManualStrategy


def reports():
    """
    Setting verbose = True will output All the statistics for each experiments
    - Statistics for Manual Strategy: Cumulative Return, Standard Deviation of Daily Returns, Mean Daily Returns
    - Statistics for Experiment 1: Cumulative Return, Standard Deviation of Daily Returns, Mean Daily Returns
    - Statistis for Experiment 2: Cumulative Return, Standard Deviation of Daily Returns, Mean Daily Returns, Sharpe Ratio
    """


    #Manual Strategy
    ManualStrategy.report(in_sample=True, verbose=False)
    ManualStrategy.report(in_sample=False, verbose=False)


    #Experiment 1
    experiment1.exp_1(verbose=False)

    #Experiment 2
    experiment2.exp_2(verbose = False)

def author():
    return 'wjo31'

if __name__ == '__main__':
    reports()
