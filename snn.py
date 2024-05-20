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
from geopy.distance import geodesic
import heapq 
import time

snn = Flask(__name__)
output_JSON_folder = 'JSON_Images'
output_folder = 'predictionImages'

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
    print("START PREDICT PROCEDURE", lat, long)

    try:
        # gather_distanced_images(lat, long)
        
        # image_from_JSON()
        
        # RUN THIS ONLY IF "Scored_Images" folder is empty
        # image_from_CSV()
        
        compare_images_in_folder()

        return "GOOD"
    except Exception as e:
        return str(e)


# def empty_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
        
def image_from_CSV():
    df = pd.read_csv('calculated_scores.csv')
    
    for index, row in df.iterrows():
        image_url = row['imageUrl']
        if pd.isna(image_url):
            print(f"No URL found for row {index}")
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
    print("START")
    try:
        
        

        finalScores = 0
        counter = 0
        
        # images = []
        for filename in os.listdir(output_folder):
            if filename.endswith('.geojson'):
        
                print("READ GEOJSON ", filename)
                filepath = os.path.join(output_folder, filename)
                with open(filepath, 'r') as file:
                    # lines = file.readlines()
                    content = file.read()
                    
                    # for line in lines:
                    try:
                        json_data = json.loads(content)
                        print("readDATA", json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error reading JSON line in {filename}: {e}")
                        continue
                
                    feature = json_data.get('features', [])[0]
                    if feature:
                        properties = feature.get('properties',{})
                        image_url = properties.get('thumb_2048_url')
                                            
                        if image_url:
                            print("URL:", image_url)
                            image = download_image(image_url)
                            if image:
                                
                                print("GOT IMAGE")
                                
                                top_scores = []
                                
                                image = image.resize((128, 128))
                                print("RESIZED")
                                
                                # image_array = preprocess_image(image)
                                for image_index, csv_image_url in enumerate(csv_image_urls):
                                    csv_image = download_image(csv_image_url)
                                    print("csvURL ",csv_image_url)
                                    if csv_image:
                                        print("GOT CSV IMAGE")
                                        
                                        # csv_image_array = preprocess_image(csv_image)
                                        csv_image = csv_image.resize((128, 128))
                                        print("CSV_RESIZED")
                                        
                                        image_array = np.array(image)
                                        csv_image_array = np.array(csv_image)


                                        # similarity_score = snn_model.predict([image_array, csv_image_array])
                                        similarity_score = snn_model.predict([np.expand_dims(image_array, axis=0), np.expand_dims(csv_image_array, axis=0)])
                                        
                                        print("TEST_URL:", image_url)
                                        print("TEST_csvURL ",csv_image_url)
                                        print("SCORE", similarity_score)
                                        print("")
                                        
                                        # print(f"Comparison score between {image_url} and {csv_image_url}: {similarity_score}")
                                        
                                        if len(top_scores) < 5:
                                            heapq.heappush(top_scores, (similarity_score, image_index))
                                        else:
                                            heapq.heappushpop(top_scores, (similarity_score, image_index))
                                            
                                
                                print("START AVERAGING")        
                                safetyScoreAve = 0 
                                for simi_score, index in top_scores:
                                    safetyScoreAve += df.loc[index, 'calculatedScores']
                                
                                safetyScoreAve /= 5
                                print("safetyScoreAve = ", safetyScoreAve)
                                finalScores += safetyScoreAve
                                counter+=1
                                print("next IMAGE")

        finalScoresAve = finalScores / counter
        print("FINISHED SAFETY SCORE CALC")
        
        return finalScoresAve
    except Exception as e:
        return str(e)
    
    
# def gather_distanced_images(lat, long):
#     try:
#         coordinates = (lat,long)
#         print("COORDINATES CONVERT",coordinates)
        
#         # Iterate over all JSON files in the MapilJSON folder
        
#         # also need to run this for pasig
        
        
#         for filename in os.listdir('Mapil_Images\paranaque'):
#             print("Checking file:", filename)
#             if filename.endswith('.geojson'):
#                 print("TEST1 - GeoJSON  file found:", filename)
#                 filepath = os.path.join('Mapil_Images\paranaque', filename)
#                 print("TEST2 - Filepath constructed:", filepath)
#                 with open(filepath, 'r') as file:
#                     json_data = json.load(file)
#                     print("TEST3 - JSON data loaded")

#                     # file_coordinates = json_data.get('coordinates')  # Assuming the key is 'coordinates'
#                     for feature in json_data.get('features', []):
#                         file_coordinates = feature['geometry'].get('coordinates')
                    
#                         if file_coordinates:
                            
#                             # print("TEST4 - Coordinates found in JSON:", file_coordinates)                     
#                             jsonCoord = (file_coordinates[1],file_coordinates[0])
#                             print("TEST5 - Coordinates found in JSON:", jsonCoord) 
                            
#                             distance = geodesic(coordinates, jsonCoord).meters
#                             print("Distance calculated:", distance)

#                         # distance = geodesic(coordinates, file_coordinates).meters
#                         # print(distance)
                        
#                             if distance < 1000:
#                                 # print("FOUND ONE")
#                                 # time.sleep(1)
#                                 if is_over_50m_from_all_images(coordinates): 
#                                     # print("SAVED")
#                                     # time.sleep(1)
                                    
#                                     output_filepath = os.path.join(output_folder, filename)
#                                     with open(output_filepath, 'w') as output_file:
#                                         json.dump(json_data, output_file)
                                
#                                     print(f"JSON data from {filename} saved in {output_folder}. Distance: {distance} meters.")
                                
#                                 # else:
#                                     # print("OVER 50m")
#                                     # time.sleep(1)
#                             else:
#                                 print(f"Distance {distance} is greater than 1000 meters.")
#                         else:
#                             print("No coordinates found in JSON data.")
#             else:
#                 print(f"{filename} is not a GeoJSON file.")    
                    
#         return "Comparison completed"
#     except Exception as e:
#         return str(e)
    
    
def gather_distanced_images(lat, long):
    try:
        coordinates = (lat, long)
        print("COORDINATES CONVERT", coordinates)

        # Iterate over all JSON files in the MapilJSON folder
        for filename in os.listdir('Mapil_Images\paranaque'):
            print("Checking file:", filename)
            if filename.endswith('.geojson'):
                print("TEST1 - GeoJSON file found:", filename)
                filepath = os.path.join('Mapil_Images/paranaque', filename)
                print("TEST2 - Filepath constructed:", filepath)
                with open(filepath, 'r') as file:
                    json_data = json.load(file)
                    print("TEST3 - JSON data loaded")

                    # Process only the first feature in the GeoJSON file
                    features = json_data.get('features', [])
                    if features:
                        feature = features[0]
                        file_coordinates = feature['geometry'].get('coordinates')
                    
                        if file_coordinates:
                            jsonCoord = (file_coordinates[1], file_coordinates[0])
                            print("TEST5 - Coordinates found in JSON:", jsonCoord)
                            
                            distance = geodesic(coordinates, jsonCoord).meters
                            print("Distance calculated:", distance)

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
                                    
                                    print(f"Relevant JSON data from {filename} saved in {output_JSON_folder}. Distance: {distance} meters.")
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
    