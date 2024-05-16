from flask import Flask, request, jsonify
from keras.models import model_from_json
import numpy as np
import json
import os

from geopy.distance import geodesic

snn = Flask(__name__)

def load_snn_model():
    with open('snn_model.json', 'r') as json_file:
        snn_model_json = json_file.read()
    snn_model = model_from_json(snn_model_json)
    snn_model.load_weights('snn_model_weights.h5')
    return snn_model


snn_model = load_snn_model()


def predict_snn(data):
    try:
        coordinates = data.get('coordinates')
        compare_coordinates_of_other_images(coordinates)
        
        
        
        # Reshape features for the model
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction
        prediction = snn_model.predict(features)
        return prediction.tolist()
    except Exception as e:
        return str(e)


def compare_coordinates_of_other_images(coordinates):
    try:
        
        output_folder = 'predictionImages'
        os.makedirs(output_folder, exist_ok=True)
        
        
        # Iterate over all JSON files in the MapilJSON folder
        for filename in os.listdir('MapilJSON'):
            if filename.endswith('.json'):
                filepath = os.path.join('MapilJSON', filename)
                with open(filepath, 'r') as file:
                    json_data = json.load(file)

                    file_coordinates = json_data.get('coordinates')  # Assuming the key is 'coordinates'
                    
                    distance = geodesic(coordinates, file_coordinates).meters
                    if distance < 1000:
                        if is_over_50m_from_all_images(coordinates, output_folder): 
                            output_filepath = os.path.join(output_folder, filename)
                            with open(output_filepath, 'w') as output_file:
                                json.dump(json_data, output_file)
                        
                            print(f"JSON data from {filename} saved in {output_folder}. Distance: {distance} meters.")
                        
        return "Comparison completed"
    except Exception as e:
        return str(e)
    
    
def is_over_50m_from_all_images(coordinates, output_folder):

    # Check distance from each image in the output folder
    for filename in os.listdir(output_folder):
        if filename.endswith('.json'):
            with open(os.path.join(output_folder, filename), 'r') as f:
                json_data = json.load(f)
            file_coordinates = json_data.get('coordinates')  # Assuming the key is 'coordinates'
            distance = geodesic(coordinates, file_coordinates).meters
            if distance < 50:
                return False
    return True


if __name__ == '__main__':
    snn.run(debug=True)
    