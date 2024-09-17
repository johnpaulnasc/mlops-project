# src/deployment.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo
model = joblib.load('models/churn_model.pkl')  # Ajuste o caminho aqui

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])
    
    # Fazer a previsão
    prediction = model.predict(data_df)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(port=5000, debug=True)