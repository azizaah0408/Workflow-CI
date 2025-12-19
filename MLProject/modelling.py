import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import pickle

# Load Data
def load_data():
    try:
        df_train = pd.read_csv('loan_data_train_processed.csv')
        df_test = pd.read_csv('loan_data_test_processed.csv')
        
        X_train = df_train.drop('Loan_Status', axis=1)
        y_train = df_train['Loan_Status']
        X_test = df_test.drop('Loan_Status', axis=1)
        y_test = df_test['Loan_Status']
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("File CSV tidak ditemukan.")
        return None, None, None, None

def train_eval_model():
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    # Ke database sqlite karena Dashboard
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Latihan Credit Scoring")
    
    mlflow.autolog()

    with mlflow.start_run():
        print("Training Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi: {acc:.4f}")
        
        with open("best_model_tuned.pkl", "wb") as f:
            pickle.dump(model, f)
            print("Model saved.")

if __name__ == "__main__":
    train_eval_model()