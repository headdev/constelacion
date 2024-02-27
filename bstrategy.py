import pandas_ta as ta

class BStrategy:
    def __init__(self, bb_len = 24, n_std= 2.0, rsi_len=14, rsi_overbought = 47, rsi_oversold= 50):
        self.bb_len = bb_len
        self.n_std = n_std
        self.rsi_len = rsi_len
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def setUp(self, df):
        bb = ta.bbands(
            close = df['close'],
            lenght = self.bb_len,
            std= self.n_std
        )

        df['lbb'] = bb.iloc[:,0]
        df['mbb'] = bb.iloc[:,1]
        df['ubb'] = bb.iloc[:,2]

        df['rsi'] = ta.rsi(close = df['close'], length=self.rsi_len)

        
        self.dataframe = df 

    def checkLonSignal(self, i= None):
        df = self.dataframe

        if i == None:
            i = len(df)

        if (df['rsi'].iloc[i] > self.rsi_overbought and df['rsi'].iloc[i] < self.rsi_oversold ): 
            # df['signal'].iloc[i] = True
            return True
        return False
    
    def checkShortSignal(self, i= None):
        df = self.dataframe

        if i == None:
            i = len(df)

        if (df['rsi'].iloc[i] < self.rsi_overbought): 
            return False
        return False


strategy = BStrategy()
