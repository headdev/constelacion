import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from twelvedata import TDClient

td = TDClient(apikey="c30ac0af06ca4533b24248fda1c28b48")

pair = 'EUR/USD'

# Construct the necessary time series
ts = td.time_series(
    symbol=pair,
    interval="1h",
    outputsize=200,
    timezone="America/New_York",
)
base = ts.as_pandas()
df_sorted = base.sort_index(ascending=True)

class BuyWithFixedSLTPStrategy(Strategy):
    def init(self):
        # Definir el stop loss y take profit como un porcentaje del precio de compra
        self.sl_distance_pips = 20  # Distancia del stop loss en pips
        self.tp_pct = 0.05  # Take Profit del 5%
        self.in_trade = False

    def next(self):
        if not self.in_trade and crossover(self.data.Close, self.data.Close[-2]):
            # Generar una señal de compra en la cruzada de medias móviles
            buy_price = self.data.Close[-1]
            sl_price = buy_price - (self.sl_distance_pips / 10000)  # Convertir pips a precio
            self.take_profit = buy_price * (1 + self.tp_pct)

            # Establecer el stop loss y take profit
            self.buy(sl=sl_price, tp=self.take_profit)
            self.in_trade = True

# Cargar tus datos históricos del par EUR/USD desde un archivo CSV
# Asegúrate de que el archivo CSV tenga una columna 'Date' y una columna 'Close'
# data = pd.read_csv('datos_eurusd.csv', parse_dates=['Date'], index_col='Date')

# Ejecutar el backtest con la estrategia definida
df_sorted = df_sorted.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open'})

bt = Backtest(df_sorted, BuyWithFixedSLTPStrategy, commission=0.002)
stats = bt.run()

# Mostrar las estadísticas del backtest
print(stats)

bt.plot()
