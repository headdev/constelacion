import sys
sys.path.insert(0, './lib')

from flip import invertir_dataframe  
from calculate_direction import CalculateDirection

import datetime
from flip import base, df_sorted 
from connect import ConnectMt5
from nystrategy import NyStrategy

launching = [[0, "23:02"]]
    
ny_open_time = datetime.datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
print(ny_open_time)

# Load NyStrategy
strategy = NyStrategy()
print(strategy.setUp(df_sorted))

for i in range(len(df_sorted)):
    print(strategy.checkLongSignal(i))

direction = CalculateDirection(df_sorted)

class Broker:
    def __init__(self, connect):
        self.connect = connect 

    def Execute(self, type = ''): 
        self.connect.run(type)

connect = ConnectMt5()
broker = Broker(connect)

while True:
    current_time = [datetime.datetime.now().weekday(), datetime.datetime.now().strftime("%H:%M")] #%S"
    print("current_time", current_time)
    if current_time in launching:
        if direction == 'Bearish':
            broker.Execute('sell')
        elif direction == 'Bullish':
            broker.Execute('buy')
        istime = True
        print("its time!", datetime.datetime.now().strftime("%H:%M:%S"))
    else: 
        istime = False
        print("is not time yet")