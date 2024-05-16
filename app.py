from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
from snn import snn_model, predict_snn


app = Flask(__name__)

# Load the model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#update when input is available
def predict_xgb(data):
    try:
        # Return prediction# Reshape features for the model
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)
        return prediction.tolist()
    except Exception as e:
        return str(e)


@app.route('/')
def home():
    return "Model Inference API: /predict_xgb for XGBoost and /predict_snn for SNN"


@app.route('/predict_xgb', methods=['POST'])
def predict_xgb_endpoint():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        # Perform prediction
        prediction = predict_xgb(data)
        # Return prediction as JSON
        return jsonify({'prediction': prediction})  # can change to just a single float value
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_snn', methods=['POST'])
def predict_snn_endpoint():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Get the image part from the JSON
        # image_link = data.get('image_link')  # Assuming the key is 'image_link'
        
        # Perform prediction using SNN model from snn.py
        prediction = predict_snn(data)  # Pass only the image link for prediction


        # Return prediction as JSON
        return jsonify({'prediction': prediction})  # or just a float value
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
