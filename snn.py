from flask import Flask, request, jsonify
import pandas as pd
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
# from geopy.distance import geodesic
import heapq 
import time

snn = Flask(__name__)
output_JSON_folder = 'JSON_Images'
output_folder = 'predictionImages'
input_folder = 'inputImage'

def load_snn_model():
    with open('new_snn_model.json', 'r') as json_file:
        snn_model_json = json_file.read()
        
    snn_model = model_from_json(snn_model_json)
    # snn_model = model_from_json(snn_model_json, custom_objects={'safe_mode': False})
    
    snn_model.load_weights('new_snn_model_weights.h5')
    
    return snn_model

snn_model = load_snn_model()

if(snn_model is not None):
    print("LOADED MODEL")

def predict_snn(lat, long):
    #print("START PREDICT PROCEDURE", lat, long)

    try:
        counter, score = get_images_in_range(lat, long)
        #print("result", counter, score)
        
        

        # gather_distanced_images(lat, long)
        # image_from_JSON()
        # RUN THIS ONLY IF "Scored_Images" folder is empty
        # image_from_CSV()
        
        input_scores = compare_images_in_folder()
        
        
        finalSafetySum = score
        for score in input_scores:
            finalSafetySum += score 
        
        finalSafetyAve = finalSafetySum / (counter + len(input_scores))
        
        #print("final count: ", counter + len(input_scores))
        #print("FINAL SUM: ", finalSafetySum)
        #print("FINAL SCORE: ", finalSafetyAve)
        
        return finalSafetyAve
    except Exception as e:
        return str(e)


def get_images_in_range(lat, long):
    counter = 0
    saved = 0
    
    score = 0
    
    # Load CSV file
    csv_file_path = 'calculated_scores_final.csv'
    data = pd.read_csv(csv_file_path)

    # Create the directory if it does not exist
    if not os.path.exists('predictionImages'):
        os.makedirs('predictionImages')

    for index, row in data.iterrows():
        # print("progress:", counter)
        coords = tuple(map(float, row['coordinates'].split(',')))
        coords = (coords[1],coords[0])
        
        distance = geodesic((lat, long), coords).meters
        # print("NEWDIST",distance)
        
        counter+=1
        if distance <= 1000:
            # print("UNDER 1000")
            image_url = row['imageUrl']
            image_path = os.path.join("predictionImages", f'image_{saved}.jpg')
            saved+=1
            
            score += row["calculatedScores"]
            
            download_image(image_url, image_path)
            # print("SAVED UNDER 1000")

    #print("finished", counter)

    return saved, score
        
def image_from_CSV():
    df = pd.read_csv('calculated_scores.csv')
    
    for index, row in df.iterrows():
        image_url = row['imageUrl']
        if pd.isna(image_url):
            #print(f"No URL found for row {index}")
            continue
        
        # Define the local image path
        image_name = f"image_{index}.jpg"  # You can modify the naming scheme as needed
        image_path = os.path.join("Scored_Images", image_name)

        # Download and save the image
        download_image(image_url, image_path)

def download_image(image_url,save_path):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)
    else:
        print(f"Failed to download image from {image_url}")
     
def image_from_JSON():
    # Iterate through all JSON files in the directory
    for filename in os.listdir("JSON_Images"):
        if filename.endswith('.geojson'):
            json_path = os.path.join("JSON_Images", filename)

            # Read the JSON file
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract the image URL
            try:
                image_url = data['features'][0]['properties']['thumb_2048_url']
            except KeyError as e:
                print(f"KeyError: {e} in file {filename}")
                continue
                
            # Define the local image path
            image_name = f"{os.path.splitext(filename)[0]}.jpg"
            image_path = os.path.join("predictionImages", image_name)

            # Download and save the image
            download_image(image_url, image_path)

    print("Image download complete.")
    return "GOOD"
           
# def preprocess_image(image):
    # # Assuming the model requires the image to be resized and converted to a numpy array
    # Modify this function based on your model's requirements
    # image = image.resize((128, 128))  # Example resize, modify as needed
    
    # # idk if this is needed
    # image_array = np.array(image)
    # image_array = np.expand_dims(image_array, axis=0)
    # return image


           
def compare_images_in_folder():
    input_folder = 'inputImage'
    scored_folder = 'Scored_Images'
    
    csv_file_path = 'calculated_scores_final.csv'
    data = pd.read_csv(csv_file_path)
    
    test_COUNTER = 0
    results = []

    for input_image in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, input_image)
        scores = []
        
        img1 = Image.open(input_image_path).convert('RGB')
        img1 = img1.resize((128, 128))
        image_array = np.array(img1)
        
        for index, scored_image in enumerate(os.listdir(scored_folder)):
            scored_image_path = os.path.join(scored_folder, scored_image)
            
            img2 = Image.open(scored_image_path).convert('RGB')
            img2 = img2.resize((128, 128))
            csv_image_array = np.array(img2)
            
            #print("TEST",test_COUNTER) 
            similarity_score = snn_model.predict([np.expand_dims(image_array, axis=0), np.expand_dims(csv_image_array, axis=0)])
            test_COUNTER+=1
        
        
            if len(scores) < 5:
                heapq.heappush(scores, (similarity_score, index))
            else:
                heapq.heappushpop(scores, (similarity_score, index))
        
        safetyScoreAve = 0 
        for score, index in scores:
            #print("SIMI", score)
            
            safe = data.loc[index, 'calculatedScores']
            
            #print("SAFETY", safe)
            
            safetyScoreAve += data.loc[index, 'calculatedScores']
        
        safetyScoreAve/=5
        
        #print("New", safetyScoreAve)    
        
        results.append(safetyScoreAve)
        
           
    return results 
    
def gather_distanced_images(lat, long):
    try:
        coordinates = (lat, long)
        #print("COORDINATES CONVERT", coordinates)

        # Iterate over all JSON files in the MapilJSON folder
        for filename in os.listdir('Mapil_Images\paranaque'):
            #print("Checking file:", filename)
            if filename.endswith('.geojson'):
                #print("TEST1 - GeoJSON file found:", filename)
                filepath = os.path.join('Mapil_Images/paranaque', filename)
                #print("TEST2 - Filepath constructed:", filepath)
                with open(filepath, 'r') as file:
                    json_data = json.load(file)
                    #print("TEST3 - JSON data loaded")

                    # Process only the first feature in the GeoJSON file
                    features = json_data.get('features', [])
                    if features:
                        feature = features[0]
                        file_coordinates = feature['geometry'].get('coordinates')
                    
                        if file_coordinates:
                            jsonCoord = (file_coordinates[1], file_coordinates[0])
                            #print("TEST5 - Coordinates found in JSON:", jsonCoord)
                            
                            distance = geodesic(coordinates, jsonCoord).meters
                            # print("Distance calculated:", distance)

                            if distance < 1000:
                                if is_over_50m_from_all_images(coordinates): 
                                    # Create a new JSON structure with only the relevant feature
                                    new_json_data = {
                                        "type": "FeatureCollection",
                                        "features": [feature]
                                    }

                                    output_filepath = os.path.join(output_JSON_folder, filename)
                                    with open(output_filepath, 'w') as output_file:
                                        json.dump(new_json_data, output_file)
                                    
                                    #print(f"Relevant JSON data from {filename} saved in {output_JSON_folder}. Distance: {distance} meters.")
                                else:
                                    print("Coordinate is not over 50m from all images.")
                            else:
                                print(f"Distance {distance} is greater than 1000 meters.")
                        else:
                            print("No coordinates found in JSON data.")
                    else:
                        print("No features found in JSON data.")
            else:
                print(f"{filename} is not a GeoJSON file.")    
                    
        return "Comparison completed"
    except Exception as e:
        return str(e)

    
def is_over_50m_from_all_images(coordinates):
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
    