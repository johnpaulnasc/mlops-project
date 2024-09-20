from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo
model = joblib.load('models/churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])
    
    # Verificar as colunas recebidas
    print(data_df.columns)  # Imprimir para ver quais colunas estão sendo enviadas
    
    # Fazer a previsão
    prediction = model.predict(data_df)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(port=5000, debug=True)