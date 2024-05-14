from flask import Flask, request, jsonify
from keras.models import model_from_json
import json
import numpy as np

snn = Flask(__name__)

@snn.route('/')
def home():
    # Load JSON and create model
    json_file = open('snn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights("snn_model_weights.h5")

    # Perform any necessary operations with the model here
    # For example, you could make predictions using the model
    
    # Return a response
    return "SNN Model Loaded Successfully!"

if __name__ == '__main__':
    snn.run(debug=True)
