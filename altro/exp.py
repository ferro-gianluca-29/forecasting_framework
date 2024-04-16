from flask import Flask, Response
import requests

app = Flask(__name__)

# Inserisci qui la tua chiave API reale
API_KEY = 'eUap1K4NFUvueAbu5qUsRroDc8wSVdSJ8N6JLkLn'

# URL dell'API per ottenere i dati
API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key=" + API_KEY + "&frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

@app.route('/metrics')
def metrics():
    # Esegui la richiesta API
    response = requests.get(API_URL)
    data = response.json()

    # Prepara l'output per Prometheus
    output = []
    for entry in data.get('response', {}).get('data', []):
        period = entry['period'].replace('-', '').replace(':', '')
        value = entry['value']  # Ottieni il valore, se non presente restituisce None
        if value is not None:
            output.append(f"eia_electricity_value{{period=\"{period}\"}} {value}")

    # Unisci tutte le metriche in una singola stringa con newline
    prometheus_output = "\n".join(output)
    return Response(prometheus_output, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
