import sys
sys.path.insert(0, './lib')

from flip import invertir_dataframe   
from  calculate_direction import CalculateDirection

# adding Folder_2 to the system path

  
class NyStrategy:
    def __init__(self, open_ny = 8, recordpath=10):
        self.open_ny = open_ny
        self.recordpath = recordpath

    def setUp(self, df):
        df['ny_time'] = 0
        df['change_percent'] = df['close'].pct_change() * 100
        df.loc[df.index.hour == 8, 'ny_time'] = 1
        self.dataframe = df

    def checkLongSignal(self, i= None):
        df = self.dataframe

        if i == None:
            i = len(df)

        if (df['ny_time'].iloc[i] == 1 ): 
            # df['signal'].iloc[i] = True
            ny_open_index = df[df['ny_time'] == 1].index[0]

            last_20_records = df.loc[ny_open_index:].tail(self.recordpath)

            base = invertir_dataframe(last_20_records)
            direction = CalculateDirection(base)
            print("Heyyy la direcion es: ", direction)
            if (direction == 'Bullish'):
                return True
        return False
    
    def checkShortSignal(self, i= None):
        df = self.dataframe

        if i == None:
            i = len(df)

        if (df['ny_time'].iloc[i] == 1 ): 
            ny_open_index = df[df['ny_time'] == 1].index[0]

            last_20_records = df.loc[ny_open_index:].tail(self.recordpath)

            base = invertir_dataframe(last_20_records)
            direction = CalculateDirection(base)
            print("Heyyy la direcion es: ", direction)
            if (direction == 'Bearish'):
                return True
        return False
