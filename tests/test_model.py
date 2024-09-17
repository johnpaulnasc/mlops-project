# tests/test_model.py

import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class TestChurnModel(unittest.TestCase):
    
    def setUp(self):
        # Carregar os dados de treino e teste
        self.train_data = pd.read_csv('../data/train_data.csv')
        self.test_data = pd.read_csv('../data/test_data.csv')
        
        # Dividir em X e y
        self.X_train = self.train_data.drop('target', axis=1)
        self.y_train = self.train_data['target']
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        
        # Instanciar e treinar o modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Salvar o modelo para os testes
        joblib.dump(self.model, '../models/test_churn_model.pkl')
    
    def test_model_accuracy(self):
        # Carregar o modelo salvo
        model = joblib.load('../models/test_churn_model.pkl')
        
        # Fazer previsões nos dados de teste
        y_pred = model.predict(self.X_test)
        
        # Calcular a acurácia
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Testar se a acurácia é maior que um limite (ex: 70%)
        self.assertGreater(accuracy, 0.7, "Acurácia do modelo é menor que 70%")

if __name__ == '__main__':
    unittest.main()