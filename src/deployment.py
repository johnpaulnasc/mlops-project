# src/deployment.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Carregar o modelo e os encoders
model = joblib.load('models/churn_model.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_contract = joblib.load('models/le_contract.pkl')
le_partner = joblib.load('models/le_partner.pkl')  
le_dependents = joblib.load('models/le_dependents.pkl')
le_internetservice = joblib.load('models/le_internetservice.pkl') 
scaler = joblib.load('models/scaler.pkl')

# Defina a ordem das features usadas no treinamento
expected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

@app.route('/', methods=['GET'])
def home():
    return "API de Previsão de Churn está rodando!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Dados recebidos: ", data)

        # Criar DataFrame a partir dos dados recebidos
        data_df = pd.DataFrame(data, index=[0])

        # Adicionar valores padrão para features ausentes
        for feature in expected_features:
            if feature not in data:
                data[feature] = 0  # Valor padrão (ou o que for adequado)

        # Reordenar o DataFrame para garantir que as colunas estejam na mesma ordem do treinamento
        data_df = data_df[expected_features]

        # Aplicar o pré-processamento (LabelEncoder, etc.)
        data_df['gender'] = le_gender.transform(data_df['gender'])
        data_df['Contract'] = le_contract.transform(data_df['Contract'])
        data_df['Partner'] = le_partner.transform(data_df['Partner'])
        data_df['Dependents'] = le_dependents.transform(data_df['Dependents'])
        data_df['InternetService'] = le_internetservice.transform(data_df['InternetService'])

        # Escalar os dados
        data_df_scaled = scaler.transform(data_df)

        # Fazer a previsão
        prediction = model.predict(data_df_scaled)
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        print("Erro: ", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)