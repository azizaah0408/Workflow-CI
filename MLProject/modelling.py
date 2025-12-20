import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import pickle
import os

# KONFIGURASI
TRAIN_PATH = 'loan_data_train_processed.csv'
TEST_PATH = 'loan_data_test_processed.csv'
OUTPUT_MODEL = 'best_model_tuned.pkl'

def load_data():
    """Load data train dan test dengan error handling sederhana"""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("❌ Error: File dataset tidak ditemukan di folder ini.")
        return None, None, None, None
        
    print("📂 Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    X_train = train.drop('Loan_Status', axis=1)
    y_train = train['Loan_Status']
    X_test = test.drop('Loan_Status', axis=1)
    y_test = test['Loan_Status']
    
    return X_train, y_train, X_test, y_test

def train_and_save():
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None:
        return

    # 2. Setup MLflow
    mlflow.autolog()

    print("🚀 Memulai Training Model...")
    with mlflow.start_run():
        # 3. Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Akurasi Model: {acc:.4f}")
        
        # 5. Simpan Model Lokal
        with open(OUTPUT_MODEL, "wb") as f:
            pickle.dump(model, f)
            
        print(f"💾 Model berhasil disimpan sebagai: {OUTPUT_MODEL}")
        
        # Log model ke MLflow (untuk artifact)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_and_save