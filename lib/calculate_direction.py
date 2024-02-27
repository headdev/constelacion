def CalculateDirection(df):
    if df['change_percent'].mean() > 0:
        tendencia = 'Bullish'
    elif df['change_percent'].mean() < 0:
        tendencia = 'Bearish'
    else:
        tendencia = 'None'

    return tendencia
