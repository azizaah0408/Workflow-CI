import pandas as pd
import pickle
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# METRICS PROMETHEUS
# 1. Menghitung jumlah request yang masuk
REQUEST_COUNT = Counter('request_count', 'Total Request Prediction', ['method', 'endpoint'])
# 2. Menghitung durasi/waktu proses
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of requests')

# LOAD MODEL
# Nama file model sesuai dengan yang ada di folder
MODEL_PATH = 'best_model_tuned.pkl' 

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None

@app.route('/')
def home():
    return "<h1>Loan Prediction API is Running! 🚀</h1><p>Use /predict endpoint to get prediction.</p>"

@app.route('/metrics')
def metrics():
    # Paksa tipe data jadi text/plain agar Prometheus mau baca
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time() # Otomatis hitung durasi waktu
def predict():
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc() # Tambah counter
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Ambil data JSON dari user
        data = request.get_json()
        
        # Ubah JSON jadi DataFrame agar mirip data training
        df = pd.DataFrame(data, index=[0])
        
        # Lakukan Prediksi
        prediction = model.predict(df)
        
        # Output (0 = Ditolak, 1 = Diterima)
        result = int(prediction[0])
        label = "Approved" if result == 1 else "Rejected"
        
        return jsonify({
            'prediction': result,
            'status': label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Jalan di port 5000
    app.run(host='0.0.0.0', port=5000)