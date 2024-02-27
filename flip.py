import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from twelvedata import TDClient
import json

td = TDClient(apikey="c30ac0af06ca4533b24248fda1c28b48")
_yesterday = 'Jueves'
_today = 'Viernes'
pair = 'EUR/USD'

# Construct the necessary time series
ts = td.time_series(
    symbol=pair,
    interval="1h",
    outputsize=24,
    timezone="America/New_York",
)

base = ts.as_pandas()


df_sorted = base.sort_index(ascending=True)

def invertir_dataframe(df):
    df_invertido = df.reset_index(drop=True)
    return df_invertido


#df_sorted
base = invertir_dataframe(base)

base['change_percent'] = base['close'].pct_change() * 100
df_sorted['change_percent'] = df_sorted['close'].pct_change() * 100


plt.title("trend")
plt.plot(df_sorted.index, df_sorted['change_percent'], marker='*', linestyle='-', color='b')
#plt.plot(df_sorted.index, df_sorted['close'], marker='*', linestyle='-', color='b')
plt.savefig('trend.png')

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(df_sorted.index, df_sorted['close'], marker='*', linestyle='-', color='b')
plt.xlabel('Index Bar')
plt.ylabel('Precio')
plt.title('Yesterday ('+_yesterday+')')
plt.grid(True)

# Subgráfico 2: Conjunto de datos resultante
plt.subplot(1, 2, 2)
plt.plot(base.index, base['close'], marker='o', linestyle='-', color='r')
plt.xlabel('Index Bar')
plt.ylabel('Precio')
plt.title('Today ('+ _today +')  + Predicted & Avg')
plt.grid(True)
plt.tight_layout()

fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

plt.savefig('./public/path-'+pair.replace('/', '')+'-'+ fecha_actual+'.png')

print(json.dumps(base['close'].to_json()))