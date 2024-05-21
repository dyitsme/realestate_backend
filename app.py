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
    
    folder_path = 'predictionImages'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete")

    print("Folder emptied successfully")
    
    # ---------------------------------------------------------------------------------------------------------
    
    folder_path = 'inputImage'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete")

    print("Folder emptied successfully")

    
    if 'coords' not in request.form:
        return "NO COORDS", 400

    if 'image' not in request.files:
        return "No image file provided", 400
    
    image_file = request.files['image']
    
    # --------------------------------------------------------------------
    coord_data = request.form['coords']
    
    # Load coordinates JSON data
    try:
        coords_dict = json.loads(coord_data)
    except json.JSONDecodeError:
        return "Invalid coordinates JSON data", 400

    # Extract latitude and longitude values
    lat = coords_dict.get('lat')
    lng = coords_dict.get('lng')
    # --------------------------------------------------------------------
    
    # Check if the file is actually an image
    if image_file.filename == '':
        return "No image selected", 400
    
    image_folder = 'inputImage'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        
    image_path = os.path.join(image_folder, image_file.filename)
    image_file.save(image_path)
    # --------------------------------------------------------------------
    
    safety_score = predict_snn(lat, lng)
    print("FINAL CALCULATED SCORE: ", safety_score,"/ 8")
    
    return "INPUT RECEIVED"

if __name__ == '__main__':
    app.run(debug=True)
