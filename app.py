from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import json
import os
import pandas as pd
from snn import snn_model, predict_snn
from nearbyamenities import nearby_amenities
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load the model
model = xgb.XGBRegressor()
model.load_model('xgb_model_gavin_best.json')

#update when input is available
def predict_xgb(parsed_data):
    data = json.loads(parsed_data)
    
    client_data = {
        'lat': data['coords']['lat'],
        'lng': data['coords']['lng'],
        'operation': data['operation'], #buy/rent
        'saleType': data['saleType'], #new, resale
        'furnishing': data['furnishing'], #unfurnished, semi, complete
        'propertyType': data['propertyType'], #house, land, etc
        'city': data['city']
    }

    df_data = {
        'bedrooms': data['bedrooms'],
        'bathrooms': data['bathrooms'],
        'lotSize': data['lotSize'],
        'floor area (m2)': data['floorArea'],
        'build (year)': data['age'],
        'total floors': data['totalFloors'],
        'car spaces': data['carSpaces'],
        "classification_Brand New": 0,
        "classification_Resale": 0,
        "fully furnished_No": 0,
        "fully furnished_Semi": 0,
        "fully furnished_Yes": 0,
        "type_encoded": 0,
        "operation_city_0_0": 0,
        "operation_city_0_1": 0,
        "operation_city_1_0": 0,
        "operation_city_1_1": 0
    }

    # List of amenities to be added to df_data
    amenities_list = [
        "gym", "wi-fi", "swimming pool", "pay tv access", "basketball court", "jogging path",
        "alarm system", "lounge", "entertainment room", "parks", "cctv", "basement parking",
        "elevators", "fire exits", "function room", "lobby", "reception area", "fire alarm",
        "fire sprinkler system", "24-hour security", "garden", "secure parking", "bar",
        "maid's room", "playground", "gymnasium", "utility room", "billiards table",
        "business center", "club house", "fitness center", "game room", "meeting rooms",
        "multi-purpose hall", "shower rooms", "sky lounge", "smoke detector", "social hall",
        "study room", "function area", "open space", "gazebos", "shops", "study area",
        "carport", "clubhouse", "deck", "gazebo", "landscaped garden", "multi-purpose lawn",
        "parking lot", "theater", "daycare center", "sauna", "laundry area", "courtyard",
        "badminton court", "tennis court", "jacuzzi", "central air conditioning", "health club",
        "indoor spa", "outdoor spa", "pool bar", "indoor pool", "drying area", "floorboards",
        "split-system heating", "garage", "remote garage", "sports facilities", "powder room",
        "maids room", "library", "spa", "clinic", "open car spaces", "intercom", "ensuite",
        "pond", "amphitheater", "gas heating", "hydronic heating", "indoor tree house",
        "open fireplace", "helipad", "golf area", "storage room", "terrace", "driver's room",
        "attic", "basement", "lanai", "ducted cooling", "ducted vacuum system", "fireplace"
    ]

    # Add amenities to df_data with initial value of 0
    for amenity in amenities_list:
        df_data[amenity] = 0

    # Update df_data based on the amenities provided in the JSON
    for amenity in data['amenities']:
        if amenity['isSelected']:
            df_data[amenity] = 1

    if client_data["propertyType"] == 'apartment':
        df_data['type_encoded'] = 0
    elif client_data['propertyType'] == 'condominium':
        df_data['type_encoded'] = 1
    elif client_data['propertyType'] == 'house':
        df_data['type_encoded'] = 2
    elif client_data['propertyType'] == 'land':
        df_data["type_encoded"] = 3
    else:   
        print("INVALID")

    if client_data['furnishing'] == 'unfurnished':
        df_data['fully furnished_No'] = 1
    elif client_data['furnishing'] == 'semi':
        df_data['fully furnished_Semi'] = 1
    elif client_data['furnishing'] == 'semi':
        df_data['fully furnished_Yes'] = 1
    else:
        print("INVALID")

    if client_data['city'] == 'pasig' and client_data['operation'] == 'buy':
        df_data['operation_city_0_1'] = 1
    elif client_data['city'] == 'pasig' and client_data['operation'] == 'rent':
        df_data['operation_city_1_1'] = 1
    elif client_data['city'] == 'paranaque' and client_data['operation'] == 'buy':
        df_data['operation_city_0_0'] = 1
    elif client_data['city'] == 'paranaque' and client_data['operation'] == 'rent':
        df_data['operation_city_1_0'] = 1
    else:
        print("INVALID")
        
    main_df = pd.DataFrame([df_data])  

    amenities = nearby_amenities(client_data['lng'], client_data['lat'])

    final_df = pd.concat([main_df, amenities], axis=1)

    try:
        # Make prediction
        prediction = model.predict(final_df)
        return prediction.tolist()
    except Exception as e:
        return str(e)


@app.route('/')
def home():
    return "Model Inference API: /predict_xgb for XGBoost and /predict_snn for SNN"

@app.route('/nearby-amenities', methods=['POST'])
def nearby_endpoint():
    nearby_amenities()

    return "DONE"

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
