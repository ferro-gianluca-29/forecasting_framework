import requests
import pandas as pd

# API URL with your personal API key
api_key = input("insert your api key:\n")
api_url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}&frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

# Perform the API request
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Extract the JSON data
    data = response.json()['response']['data']
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv('./Energy_Data.csv', index=False)
    
    print("CSV file successfully saved.")
else:
    print("API request error: Status", response.status_code)

