
from TheoreticallyOptimalStrategy import TheoreticallyOptimalStrategy, result
import datetime as dt
from indicators import sma, ema, macd, tsi, bollinger_band, run
import warnings

warnings.filterwarnings('ignore')

sd = dt.datetime(2010, 1, 1)
ed = dt.datetime(2011,12,31)
symbol = 'JPM'

#Initializing the object
tos = TheoreticallyOptimalStrategy()
df_trades = tos.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)

#Content of the report - Comparison Chart and Metrics
result()

#Saving Indicator Charts into Image folder
run(sd, ed, symbol)

def author():
    return 'wjo31'