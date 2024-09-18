# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    # Carregar o dataset
    data = pd.read_csv(input_path)
    
    # Remover valores ausentes
    data = data.dropna()
    
    # Remover colunas irrelevantes (ex: 'customerID')
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
    
    # Codificar variáveis categóricas
    labelencoder = LabelEncoder()
    
    # Codificar todas as colunas categóricas
    for column in data.columns:
        if data[column].dtype == 'object':  # Verifica se a coluna é do tipo 'object' (string/categórica)
            data[column] = labelencoder.fit_transform(data[column])
    
    # Definir a coluna alvo (ex: 'Churn')
    target = 'Churn'
    
    # Features e labels
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar os dados numéricos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Salvar os dados pré-processados
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    train_data['target'] = y_train
    test_data['target'] = y_test
    
    train_data.to_csv(output_path + 'train_data.csv', index=False)
    test_data.to_csv(output_path + 'test_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data('data/raw_data.csv', 'data/')