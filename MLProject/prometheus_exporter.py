import pandas as pd
import pickle
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest

app = Flask(__name__)

# METRICS
REQUEST_COUNT = Counter('request_count', 'Total Request Prediction', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of requests')

# LOAD MODEL
MODEL_PATH = 'best_model_tuned.pkl' 

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    model = None

@app.route('/')
def home():
    return "<h1>Loan Prediction API is Running!</h1>"

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        prediction = model.predict(df)
        result = int(prediction[0])
        label = "Approved" if result == 1 else "Rejected"
        
        return jsonify({'prediction': result, 'status': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)