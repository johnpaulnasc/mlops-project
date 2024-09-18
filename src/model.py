# src/model.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    
    # Aplicar codificação de variáveis categóricas
    le_gender = LabelEncoder()
    le_contract = LabelEncoder()
    
    # Codificar colunas categóricas específicas
    X_train['gender'] = le_gender.fit_transform(X_train['gender'])
    X_train['Contract'] = le_contract.fit_transform(X_train['Contract'])
    
    # Salvar os encoders para uso na previsão
    joblib.dump(le_gender, model_output_path + 'le_gender.pkl')
    joblib.dump(le_contract, model_output_path + 'le_contract.pkl')
    
    # Aplicar escalonamento de variáveis numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Salvar o escalador
    joblib.dump(scaler, model_output_path + 'scaler.pkl')

    # Instanciar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treinar o modelo
    model.fit(X_train_scaled, y_train)
    
    # Verificar se a pasta de saída existe, e criar se necessário
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # Salvar o modelo treinado
    joblib.dump(model, model_output_path + 'churn_model.pkl')
    print("Modelo e encoders salvos com sucesso!")

if __name__ == "__main__":
    train_model('data/train_data.csv', 'models/')