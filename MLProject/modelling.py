import requests
import pandas as pd
import json

# URL ke Server (Servernya adalah file prometheus_exporter.py)
url = 'http://127.0.0.1:5000/predict'

try:
    print("[Client] Mengirim request...")
    df_test = pd.read_csv('loan_data_test_processed.csv')
    sample_data = df_test.sample(1)
    if 'Loan_Status' in sample_data.columns:
        sample_data = sample_data.drop('Loan_Status', axis=1)
    
    data_to_send = sample_data.to_dict(orient='records')[0]
    
    response = requests.post(url, json=data_to_send)
    
    print("--- HASIL PREDIKSI ---")
    print(response.json())

except Exception as e:
    print(f"Error: {e}")