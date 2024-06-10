from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from notebooks.utils.funcs import transform_x

# Cargar el modelo preentrenado
# model = joblib.load('models/best_model/best_model.pkl')
model = joblib.load('models/logit/logit_model_trained.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def process_data():
    if request.is_json:

        # try:
        # Recibir JSON
        input_data = request.get_json()

        # Procesar los datos y transformarlos para que el modelo lo entienda        
        input_data_df = pd.DataFrame([input_data])
        input_data_df = transform_x(input_data_df)

        # Realizar la predicción
        prediction = model.predict(input_data_df)
        prediction = 'Nuevo' if prediction[0] == 1 else 'Usado'

        # Calcular la probabilidad
        proba = model.predict_proba(input_data_df)
        proba = proba[0, 1]

        # Procesar los datos (este es solo un ejemplo)
        prediction_r = {
            "prediction": prediction, 
            "probability": proba
        } #, "received_data": input_data}
        
        # Devolver la prediccion
        return jsonify(prediction_r)

        # except Exception as e:
        #     return jsonify({"error": str(e)}), 500
        
    else:
        return jsonify({"error": "Invalid input, JSON expected"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)