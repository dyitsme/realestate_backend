# from flask import Flask, request, jsonify
# import json
# import numpy as np
# import xgboost as xgb

# app = Flask(__name__)

# # Load the XGBoost model from JSON file
# def load_model(model_path):
#     with open(model_path, 'r') as f:
#         model_json = json.load(f)
#     model = xgb.Booster(model_file=None)
#     model.load_model(model_json)
#     return model

# model = load_model('your_model.json')

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

# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     # Get input data from request
#     input_data = request.json
#     # Make predictions
#     predictions = predict(input_data)
#     # Return the predictions as JSON response
#     return jsonify(predictions.tolist())

# if __name__ == '__main__':
#     app.run(debug=True)
