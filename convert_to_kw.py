import pandas as pd

# Carica i dati dal file CSV
data = pd.read_csv("./data/Dataset/Consumi.csv", delimiter=';')

# Sostituisci la virgola con il punto e converti le colonne in float
data = data.apply(lambda x: x.str.replace(',', '.').astype(float))

# Moltiplica ogni valore per 4 (conversione kWh in kW ogni 15 minuti)
data *= 4

# Salva i nuovi dati in un nuovo file CSV
data.to_csv('Consumer_power.csv', index=False)
