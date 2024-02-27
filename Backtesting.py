from backtesting import Backtest, Strategy
from backtesting.test import EURUSD, SMA

from twelvedata import TDClient
from backtesting.test import GOOG

td = TDClient(apikey="c30ac0af06ca4533b24248fda1c28b48")

pair = 'EUR/USD'

# # Construct the necessary time series
ts = td.time_series(
    symbol=pair,
    interval="1h",
    outputsize=95,
    timezone="America/New_York",
)
df = ts.as_pandas()
df = df.sort_index(ascending=True)

# df = EURUSD
df['ny_time'] = 0

print(df.columns)

df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open'})
df.loc[df.index.hour == 8, 'ny_time'] = 1
df.loc[df.index.hour == 15, 'ny_time'] = -1

count_nytime_1 = (df['ny_time'] == -1).sum()




import numpy as np

class Ny(Strategy):
    n1 = 10
    n2 = 20
    
    def init(self):
        self.tp_distance_pips = 50
        self.sl_distance_pips = 20
        close = self.data.Close

    def next(self):
        high, low, close = self.data.High, self.data.Low, self.data.Close
        
        price_delta = .0050
        upper, lower = close[-1] * (1 + np.r_[1, -1] * price_delta)
    
        current_signal = self.data.ny_time[-1]

        if current_signal == 1:
            price = self.data.Close[-1]

            print('at:', price,  'sl_price', lower, 'tp', upper)

            # Establecer el stop loss y take profit
            if not self.position:
                self.buy(sl=round(lower, 5), tp=round(upper, 5))
                #self.buy(size=1.0, limit=None, sl=lower,tp=upper)
        
        elif current_signal == -1:
            if self.position:
                self.position.close()



bt = Backtest(EURUSD, Ny,
              cash=10_000, commission=.02)

output = bt.run()
print(output)

bt.plot()