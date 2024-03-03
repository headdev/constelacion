def calculate_lot_size(stop_loss, risk_percentage):
    account_balance = 932
      # Supongamos que el saldo de la cuenta es de 10000
    risk = risk_percentage * account_balance
    
    valpip = risk / stop_loss

    print('valpip', valpip, 'risk', risk/100)
    lot_size = (float(account_balance) * (float(risk_percentage)/100)) / (valpip * 10)
    # lot_size = valpip #/ 0.10
    
    return lot_size #/ contract_size

# Ejemplo de uso
stop_loss = 10  # Stop-loss en pips
risk_percentage = 0.01  # 1% de riesgo
lot_size = calculate_lot_size(stop_loss, risk_percentage)
print("Tamaño del lote:", lot_size)


