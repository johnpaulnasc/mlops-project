import os
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class TestChurnModel(unittest.TestCase):
    
    def setUp(self):
        # Obter o caminho absoluto para o arquivo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_data = pd.read_csv(os.path.join(base_dir, '../data/train_data.csv'))
        self.test_data = pd.read_csv(os.path.join(base_dir, '../data/test_data.csv'))
        
        # Dividir em X e y
        self.X_train = self.train_data.drop('target', axis=1)
        self.y_train = self.train_data['target']
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        
        # Remover valores NaN de y_train
        if self.y_train.isnull().sum() > 0:
            print(f"Removendo {self.y_train.isnull().sum()} valores nulos de y_train.")
            valid_indices_train = self.y_train.dropna().index
            self.X_train = self.X_train.loc[valid_indices_train]
            self.y_train = self.y_train.dropna()
        
        # Remover valores NaN de y_test
        if self.y_test.isnull().sum() > 0:
            print(f"Removendo {self.y_test.isnull().sum()} valores nulos de y_test.")
            valid_indices_test = self.y_test.dropna().index
            self.X_test = self.X_test.loc[valid_indices_test]
            self.y_test = self.y_test.dropna()
        
        # Instanciar e treinar o modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Salvar o modelo para os testes
        joblib.dump(self.model, os.path.join(base_dir, '../models/test_churn_model.pkl'))
    
    def test_model_accuracy(self):
        # Carregar o modelo salvo
        model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/test_churn_model.pkl'))
        
        # Fazer previsões nos dados de teste
        y_pred = model.predict(self.X_test)
        
        # Calcular a acurácia
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Testar se a acurácia é maior que um limite (ex: 70%)
        self.assertGreater(accuracy, 0.7, "Acurácia do modelo é menor que 70%")

if __name__ == '__main__':
    unittest.main()
