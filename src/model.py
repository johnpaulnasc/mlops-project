import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(train_data_path, model_output_path):
    # Carregar os dados de treino
    train_data = pd.read_csv(train_data_path)
    
    # Verificar se há NaNs em y_train
    if train_data['target'].isnull().sum() > 0:
        print(f"Há {train_data['target'].isnull().sum()} valores nulos na coluna target. Removendo esses valores...")
        train_data = train_data.dropna(subset=['target'])

    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    
    # Instanciar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Verificar se a pasta de saída existe, e criar se necessário
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # Salvar o modelo treinado
    joblib.dump(model, model_output_path + 'churn_model.pkl')
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    train_model('data/train_data.csv', 'models/')