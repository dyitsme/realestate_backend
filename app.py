from flask import Flask, request, jsonify
import json
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#update when input is available
def predict(data):
    try:
        redict(features)
        # Return prediction# Reshape features for the model
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = model.p
        return prediction.tolist()
    except Exception as e:
        return str(e)



@app.route('/')
def home():
    return "XGBoost Model Inference API"


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        # Extract features
        features = data['features']
        # Perform prediction
        prediction = predict(features)
        # Return prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    


# @app.route('/load_model', methods=['GET'])
# @app.route('/load_model')

# # Load the XGBoost model from JSON file
# def load_model(model_path):
#     with open(model_path, 'r') as f:
#         model_json = json.load(f)
#     model = xgb.Booster(model_file=None)
#     model.load_model(model_json)
#     return model

# model = load_model('xgb_model.json')

    
# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     # Get input data from request
#     input_data = request.json
#     # Make predictions
#     predictions = predict(input_data)
#     # Return the predictions as JSON response
#     return jsonify(predictions.tolist())

# # Preprocess input data
# def preprocess_data(data):
#     # Your preprocessing steps here
#     # Ensure the input data is in the right format
#     return data

# # Make predictions
# def predict(data):
#     # Preprocess the data
#     preprocessed_data = preprocess_data(data)
#     # Convert the preprocessed data into DMatrix format
#     dmatrix = xgb.DMatrix(np.array(preprocessed_data))
#     # Use the loaded model to make predictions
#     predictions = model.predict(dmatrix)
#     return predictions


if __name__ == '__main__':
    app.run(debug=True)
