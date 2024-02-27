import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_balance = 10000, leverage = 10, trailing_stop_loss = False):
        self.inicial_balance = initial_balance
        self.balance = initial_balance
        self.amount = 0
        self.leverage = 10
        self.fee_cost = 0.02/100
        self.inv = self.balance * 0.01
        print("Risk", self.inv) 
        self.profit = []
        self.drawdown = []

        self.winned = 0
        self.lossed = 0

        self.num_operations = 0
        self.num_longs = 0
        self.num_shorts = 0

        self.tp_distance_pips = 20
        self.sl_distance_pips = 20

        self.is_long_open = False
        self.is_short_open = False

    def open_position(self, price, side, from_opened = 0):
        self.num_operations += 1
        if side == 'long':   
            self.num_longs += 1 

            if self.is_long_open:
                self.is_long_open = True
                self.amount += self.inv/price
            else:
                self.is_long_open = True
                print('[open long]', self.is_long_open)
                self.long_open_price = price
                self.amount = self.inv/price

        elif side == 'short':
            self.num_shorts += 1

            if self.is_short_open:
                self.short_open_price += (self.short_open_price + price ) / 2 
                self.amount += self.inv/price 
            else:
                self.is_short_open = True 
                self.short_open_price= price
                self.amount = self.inv/price 
        # self.amount = self.inv / price         

    def close_position(self, price, type='sl',  index=0):
        self.num_operations += 1
        # result = 0
        if self.is_long_open:
            if (type=='sl'):
                result = -1 * self.amount 
            elif type=='tp':
                result = self.amount 

            print("result operation:", result)
            self.is_long_open = False
            self.long_open_price = 0

        elif self.is_short_open:
            if (type=='sl'):
                result = -1 * self.amount
            elif type=='tp':
                result = self.amount
            
            
            self.is_short_open = False
            self.short_open_price = 0
        
        print("result operation Short:", result, 'type', type)

        self.profit.append(result)
        self.balance += result

        if result > 0:
            self.winned += 1
            self.drawdown.append(0)
        else:
            self.lossed +=1
            self.drawdown.append(result)
        
        self.take_profit_price = 0
        self.stop_loss_price = 0
        
        
    def get_tp(self):
        return self.take_profit_price

    def set_tp(self, price):
        if self.is_long_open:
            _tp = price + (self.tp_distance_pips / 10000)
            self.take_profit_price = round(_tp, 5) 

        elif self.is_short_open:
            _tp = price - (self.tp_distance_pips / 10000)
            self.take_profit_price = round(_tp, 5)

    def get_sl(self):
        return self.stoploss_price
    
    def set_sl(self, price):

        if self.is_long_open:
            _sl = price - (self.sl_distance_pips / 10000)
            self.stoploss_price = round(_sl, 5) 

        elif self.is_short_open:
            _sl = price + (self.sl_distance_pips / 10000)
            self.stoploss_price = round(_sl , 5) 

    def reset_results(self):
        self.balance = self.initial_balance
        self.amount = 0

        self.profit = []
        self.drawdown = []
        self.winned = 0
        self.lossed = 0
        
        self.num_operations = 0
        self.num_longs = 0
        self.num_shorts = 0

        self.is_long_open = False
        self.is_short_open = False


    def return_result(self, symbol, start_date, end_date):
        profit = sum(self.profit)
        drawdown = sum(self.drawdown)

        fees = (abs(profit) * self.fee_cost * self.num_operations)
        results = {
            'symbol': symbol,
            'start': start_date,
            'end': end_date,
            'balance': self.balance,
            'profit' : profit,
            'drawdown': drawdown,
            'profit_after_fees': round(profit - fees, 2),
            'num_operations': self.num_operations,
            'num_longs': self.num_longs,
            'num_shorts': self.num_shorts,
            'winned': self.winned,
            'lossed': self.lossed
        }

        if self.num_operations > 0:
            winrate = self.winned / self.num_operations
            results['winrate'] = winrate
            results['fitness_function'] = ((profit- abs(drawdown)) * winrate) / self.num_operations

        else:
            results['winrate'] = 0
            results['fitness_function'] = 0

        return results


    def show (self):

        plt.plot(self.profit, label='Profit')

        # Agregar etiquetas y leyenda
        plt.xlabel('Time')
        plt.ylabel('Profit')
        plt.title('Profit')
        plt.legend()

        # Mostrar el gráfico
        plt.plot(self.drawdown, label='Dd')

        # Agregar etiquetas y leyenda
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.title('Drawdown')
        plt.legend()

        # Mostrar el gráfico
        plt.show()

    def infoPosition(self):
        if (self.is_long_open):
            at = self.long_open_price
        elif self.is_short_open:
            at = self.short_open_price

        return {'sl': self.get_sl(), 'tp': self.get_tp(), 'at': at}
    

    def __backtesting__(self, df, strategy):
        high = df['high']
        close = df['close']
        low = df['low']

        for i in range(len(df)):
            if self.balance > 0:

                if strategy.checkLongSignal(i):
                    df['signal'].iloc[i] = 1
                    self.open_position(price=close[i], side= 'long', from_opened= i)
                    self.set_tp(price = close[i])
                    self.set_sl(price = close[i])

                    print(self.infoPosition())
                elif strategy.checkShortSignal(i):
                    df['signal'].iloc[i] = -1
                    self.open_position(price= close[i], side ='short', from_opened = i)
                    self.set_tp(price = close[i])
                    self.set_sl(price = close[i])
                    print(self.infoPosition())
                else:

                    if self.is_long_open:
                        if high[i] >= self.get_tp():
                            self.close_position(price = self.get_tp(), type='tp', index=i)
                            print('touch TP', self.get_tp(), 'priceUp', high[i] )
                        elif low[i] <= self.get_sl():
                            self.close_position(price = self.get_sl(), type='sl')
                    
                    elif self.is_short_open:
                        if high[i] >= self.get_sl():
                            print('touch sl', self.get_sl(), 'pricebroke', high[i] )
                            self.close_position(price = self.get_sl(), type='sl')
                        elif low[i] <= self.get_tp():
                            self.close_position(price = self.get_tp(), type='tp')
                        

from nystrategy import NyStrategy

from twelvedata import TDClient

td = TDClient(apikey="c30ac0af06ca4533b24248fda1c28b48")
_yesterday = 'Jueves'
_today = 'Viernes'
pair = 'EUR/USD'

# Construct the necessary time series
ts = td.time_series(
    symbol=pair,
    interval="1h",
    outputsize=72,
    timezone="America/New_York",
)
base = ts.as_pandas()

base.plot()
df_sorted = base.sort_index(ascending=True)

strategy = NyStrategy()
strategy.setUp(df_sorted)


tryback = Backtester()
df_sorted['signal'] = 0
tryback.__backtesting__(df_sorted, strategy)


df_sorted = df_sorted.reset_index(drop=True)
ny_open_index = df_sorted[df_sorted['signal'] == -1].index
print("indice", ny_open_index)

print(df_sorted)
plt.plot(df_sorted.index, df_sorted['close'], label='Live')
plt.legend()

# Si el índice donde ocurre la apertura de NY a las 8 a.m. está en el rango de los datos, graficamos un triángulo
# if ny_open_index in df_sorted.index:

for nyidx in ny_open_index:
    plt.axvline(x=nyidx, color='r', linestyle='--', label='Apertura de NY')

plt.show()

print(tryback.return_result(symbol = 'eurusd', start_date='-', end_date='-'))

