import pandas as pd

# Carica il tuo DataFrame (assumendo che la prima colonna contenga le date)
df = pd.read_csv("./data/Dataset/PV_marocco.csv", sep=';')

# Converti la prima colonna in datetime, specificando che il giorno viene prima del mese
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], dayfirst=True)

# Imposta la prima colonna come indice
df.set_index(df.columns[0], inplace=True)

# Controlla e gestisci gli indici duplicati
if df.index.duplicated().any():
    print("Attenzione: ci sono indici duplicati. Verranno rimossi.")
    df = df[~df.index.duplicated(keep='first')]  # Mantiene solo la prima occorrenza

# Assegna una frequenza temporale di 5 minuti e riempie i valori mancanti
df = df.resample('5min').asfreq()

# Setta i valori a zero fuori dall'orario 7:00-18:00
mask_outside_hours = (df.index.time < pd.to_datetime('07:00').time()) | (df.index.time > pd.to_datetime('18:00').time())
df.loc[mask_outside_hours] = 0

# Interpola i valori NaN usando il metodo 'time'
df.interpolate(method='time', inplace=True)

# Prepara una maschera per i valori zero all'interno dell'orario 7:00-18:00
mask_inside_hours = ~mask_outside_hours
df_zero_inside = (df == 0) & mask_inside_hours.reshape(-1, 1)  # Utilizza reshape direttamente

# Imposta i valori zero identificati a NA per consentire l'interpolazione
df.mask(df_zero_inside, other=pd.NA, inplace=True)

# Interpola nuovamente i valori NA usando il metodo 'time'
df.interpolate(method='time', inplace=True)

# Tronca i valori a tre cifre decimali
df = df.round(3)

# Salva il DataFrame modificato in un nuovo file CSV
df.to_csv("PV_marocco_preprocessed.csv")
