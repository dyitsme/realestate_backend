from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import json
import os
from snn import snn_model, predict_snn

from werkzeug.utils import secure_filename


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
    
    print("TEST")

    # expect point data
    
    if 'coords' not in request.form:
        return "NO COORDS", 400

    if 'image' not in request.files:
        return "No image file provided", 400
    
    image_file = request.files['image']
    
    coord_data = request.form['coords']
    
    print("COORDS",coord_data)
    
    # Load coordinates JSON data
    try:
        coords_dict = json.loads(coord_data)
    except json.JSONDecodeError:
        return "Invalid coordinates JSON data", 400

    # Extract latitude and longitude values
    lat = coords_dict.get('lat')
    lng = coords_dict.get('lng')
    
    print("LATLONG", lat,lng)
    
    # Check if the file is actually an image
    if image_file.filename == '':
        return "No image selected", 400
    
    print("Image received and processed successfully")
    
    
    predict_snn(image_file, lat, lng)
    
    return "INPUT RECEIVED"

if __name__ == '__main__':
    app.run(debug=True)
