# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model(test_data_path, model_path):
    # Carregar os dados de teste
    test_data = pd.read_csv(test_data_path)
    
    # Verificar se há NaNs na coluna target
    if test_data['target'].isnull().sum() > 0:
        print(f"Há {test_data['target'].isnull().sum()} valores nulos na coluna target. Removendo esses valores...")
        test_data = test_data.dropna(subset=['target'])
    
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Carregar o modelo treinado
    model = joblib.load(model_path + 'churn_model.pkl')
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar o modelo
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model('data/test_data.csv', 'models/')