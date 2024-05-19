from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.utils import get_custom_objects
import numpy as np
import shutil
import json
import os
import requests
from PIL import Image
from io import BytesIO
from geopy.distance import geodesic
import heapq 

snn = Flask(__name__)
output_folder = 'predictionImages'

def load_snn_model():
    with open('new_snn_model.json', 'r') as json_file:
        snn_model_json = json_file.read()
        
    snn_model = model_from_json(snn_model_json)
    # snn_model = model_from_json(snn_model_json, custom_objects={'safe_mode': False})
    
    snn_model.load_weights('new_snn_model_weights.h5')
    
    return snn_model

# def load_snn_model():
#     # Load the model architecture
#     with open('snn_model.json', 'r') as json_file:
#         snn_model_json = json_file.read()
    
#     custom_objects = {'Functional': tf.keras.models.Model}

#     snn_model = model_from_json(snn_model_json, custom_objects=custom_objects)
    
#     # Recreate the model from JSON
#     # snn_model = model_from_json(snn_model_json)
    
#     # Load weights into the model
#     snn_model.load_weights('snn_model_weights.h5')
    
#     # Compile the model
#     snn_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
#                       optimizer=tf.keras.optimizers.Adam(), 
#                       metrics=['accuracy'])
#     return snn_model

snn_model = load_snn_model()

if(snn_model is not None):
    print("LOADED MODEL")

def predict_snn(image, lat, long):
    print("START PREDICT PROCEDURE", lat, long)

    try:
        compare_coordinates_of_other_images(image, lat, long)
        
        # continue the calculation here 
        # output_folder = 'predictionImages'
        compare_images_in_folder(output_folder)

        return 
    except Exception as e:
        return str(e)


def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        
 

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        img = Image.open(BytesIO(response.content))
        return img
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None 

           
# def preprocess_image(image):
    # # Assuming the model requires the image to be resized and converted to a numpy array
    # Modify this function based on your model's requirements
    # image = image.resize((128, 128))  # Example resize, modify as needed
    
    # # idk if this is needed
    # image_array = np.array(image)
    # image_array = np.expand_dims(image_array, axis=0)
    # return image


           
def compare_images_in_folder(folder):
    try:
        df = pd.read_csv('calculated_scores.csv')
        csv_image_urls = df['imageUrl'].tolist()

        finalScores = 0
        counter = 0
        
        # images = []
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r') as file:
                    json_data = json.load(file)
                    image_url = json_data.get('imageUrl')  # Assuming the key is 'imageUrl'
                    if image_url:
                        image = download_image(image_url)
                        if image:
                            top_scores = []
                            
                            image = image.resize(128, 128)
                            # image_array = preprocess_image(image)
                            for image_index, csv_image_url in enumerate(csv_image_urls):
                                csv_image = download_image(csv_image_url)
                                if csv_image:
                                    # csv_image_array = preprocess_image(csv_image)
                                    csv_image = csv_image.resize(128, 128)

                                    similarity_score = snn_model.predict([image, csv_image])
                                    # print(f"Comparison score between {image_url} and {csv_image_url}: {comparison_score}")
                                    
                                    if len(top_scores) < 5:
                                        heapq.heappush(top_scores, (similarity_score, image_index))
                                    else:
                                        heapq.heappushpop(top_scores, (similarity_score, image_index))
                                        
                                    safetyScoreAve = 0 
                                    for simi_score, index in top_scores:
                                        safetyScoreAve += df.loc([index, 'calculatedScores'])
                                    
                                    safetyScoreAve /= 5
                                    finalScores += safetyScoreAve
                                    counter+=1

        finalScoresAve = finalScores / counter
        print("FINISHED SAFETY SCORE CALC")
        
        return finalScoresAve
    except Exception as e:
        return str(e)
    
    
def compare_coordinates_of_other_images(image, lat, long):
    try:
        
        # output_folder = 'predictionImages'
        
        if os.path.exists(output_folder):
            empty_folder(output_folder)
        
        os.makedirs(output_folder, exist_ok=True)
        
        
        # include the input image in the folder
        
        
        # Iterate over all JSON files in the MapilJSON folder
        for filename in os.listdir('Mapil_Images'):
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
    