# src/train_model.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(train_data_path):
    # Iniciar o rastreamento do experimento no MLflow
    mlflow.start_run()
    
    # Carregar os dados
    data = pd.read_csv(train_data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Instanciar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Avaliar o modelo
    accuracy = model.score(X_test, y_test)
    
    # Registrar as métricas no MLflow
    mlflow.log_metric("accuracy", accuracy)
    
    # Registrar o modelo treinado no MLflow
    mlflow.sklearn.log_model(model, "model")

    # Salvar o modelo localmente
    joblib.dump(model, 'models/churn_model.pkl')

    # Finalizar o rastreamento do experimento
    mlflow.end_run()

if __name__ == "__main__":
    train_model('data/train_data.csv')