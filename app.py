import shutil
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import json
import os
import pandas as pd
from snn import snn_model, predict_snn
from nearbyamenities import nearby_amenities
from werkzeug.utils import secure_filename
from shapely.geometry import Point
import geopandas as gpd
from shapely.ops import nearest_points
from flask_cors import CORS
import shap
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = xgb.XGBRegressor()
model.load_model('xgb_model_gavin_best_96.json')

# Load the scaler
amenity_scaler = joblib.load('amenity_robust_scaler.pkl')

flood_scaler = joblib.load('flood_robust_scaler.pkl')

@app.route('/')
def home():
    return "Model Inference API: /predict_xgb for XGBoost and /predict_snn for SNN"

@app.route('/nearby-amenities', methods=['POST'])
def nearby_endpoint():
    nearby_amenities()

    return "DONE"

# @app.route('/predict_xgb', methods=['POST'])
# def predict_xgb_endpoint():
#     try:
#         # Get JSON data from request
#         data = request.form.to_dict()
#         # Perform prediction
#         prediction = predict_xgb(data)
#         # Return prediction as JSON
#         return jsonify({'prediction': prediction})  # can change to just a single float value
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# Function to calculate the distance from a point to the nearest fault line
def min_distance_to_fault(point, fault_lines):
    try:
        nearest_line = nearest_points(point, fault_lines.unary_union)[1]
        return point.distance(nearest_line)
    except Exception as e:
        print(f"Error calculating minimum distance: {e}")
        return None
    

# Function to get expected feature names from XGBoost model
def get_expected_features(model):
    try:
        # Assuming model is an XGBRegressor or XGBClassifier
        return model.get_booster().feature_names
    except Exception as e:
        print(f"Error getting feature names: {str(e)}")
        return []
    
@app.route('/predict_xgb', methods=['POST'])
def predict_xgb_endpoint():
    try:
        #coordinates
        coords_str = request.form.get('coords')
        coords_data = json.loads(coords_str)

        #operation
        op_str = request.form.get('operation')
        # op_data = json.loads(op_str)

        #saleType
        st_str = request.form.get('saleType')
        # st_data = json.loads(st_str)

        #furnish
        furnish_str = request.form.get('furnishing')
        # furnish_data = json.loads(furnish_str)

        #propertyType
        pt_str = request.form.get('propertyType')
        # pt_data = json.loads(pt_str)

        #city
        city_str = request.form.get('city')
        # city_data = json.loads(city_str)

        amenities_str = request.form.get('amenities')

        # Safely parse the string into a Python list
        data_list = ast.literal_eval(amenities_str)

        # Initialize an empty list to store split words
        split_words = []

        # Split each element in the list into words
        for item in data_list:
            words = item.split(',')  # Split by whitespace
            split_words.extend(words)  # Extend the list with split words

        # Print the resulting list of split words
        print(split_words)

        #amenities_data = json.loads(amenities_str)
        #print(amenities_data)

        client_data = {
            'lat': coords_data['lat'],
            'lng': coords_data['lng'],
            'operation': op_str, #buy/rent
            'saleType': st_str, #new, resale
            'furnishing': furnish_str, #unfurnished, semi, complete
            'propertyType': pt_str, #house, land, etc
            'city': city_str,
            'amenities': split_words
        }

        #bedrooms
        bedrooms_str = request.form.get('bedrooms')
        # bedrooms = json.loads(bedrooms_str)

        # bathrooms
        bathrooms_str = request.form.get('bathrooms')
        # bathrooms = json.loads(bathrooms_str)

        # lotSize
        lot_size_str = request.form.get('lotSize')
        # lot_size = json.loads(lot_size_str)

        # floor area (m2)
        floor_area_str = request.form.get('floorArea')
        # floor_area = json.loads(floor_area_str)

        # build (year)
        build_year_str = request.form.get('age')
        # build_year = json.loads(build_year_str)

        # total floors
        total_floors_str = request.form.get('totalFloors')
        # total_floors = json.loads(total_floors_str)

        # car spaces
        car_spaces_str = request.form.get('carSpaces')
        # car_spaces = json.loads(car_spaces_str)

        df_data = {
            'bedrooms': bedrooms_str,
            'bathrooms': bathrooms_str,
            'floor area': floor_area_str,
            'land size': lot_size_str,
            'build (year)': build_year_str,
            'total floors': total_floors_str,
            'car spaces': car_spaces_str,
            'rooms (total)': bedrooms_str,
            "classification_Brand New": False,
            "classification_Resale": False,
            "fully furnished_No": False,
            "fully furnished_Semi": False,
            "fully furnished_Yes": False,
            "type_encoded": 0,
            "operation_city_0_0": False,
            "operation_city_0_1": False,
            "operation_city_1_0": False,
            "operation_city_1_1": False,
            "ModelScores": 0,
            "sqm_upper": floor_area_str,
            "sqm_lower": floor_area_str,
            "min_distance_to_fault_meters": 0,
            "flood_threat_level_5_yr": 0,
            "flood_threat_level_25_yr": 0
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
            "attic", "basement", "lanai", "ducted cooling", "ducted vacuum system", "fireplace",
            "broadband internet available", "built-in wardrobes", "baths", "fully fenced",
            "air conditioning", "balcony", 'dryer', 'dryer.1', 'duct', 'duct.1', 'fibre', "living room"
        ]

        # Add amenities to df_data with initial value of 0
        for amenity in amenities_list:
            df_data[amenity] = 0

        # Update df_data based on the amenities provided in the JSON
        for amenity in client_data['amenities']:
           df_data[amenity] = 1
           print(amenity)

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

        if client_data['saleType'] == 'resale':
            df_data['classification_Resale'] = True
        elif client_data['saleType'] == 'new':
            df_data['classification_Brand New'] = True
        else:
            print("INVALID")

        if client_data['furnishing'] == 'complete':
            df_data['fully furnished_No'] = True
        elif client_data['furnishing'] == 'semi':
            df_data['fully furnished_Semi'] = True
        elif client_data['furnishing'] == 'unfurnished':
            df_data['fully furnished_Yes'] = True
        else:
            print("INVALID")

        if client_data['city'] == 'Pasig' and client_data['operation'] == 'buy':
            df_data['operation_city_0_1'] = True
        elif client_data['city'] == 'Pasig' and client_data['operation'] == 'rent':
            df_data['operation_city_1_1'] = True
        elif client_data['city'] == 'Parañaque' and client_data['operation'] == 'buy':
            df_data['operation_city_0_0'] = True
        elif client_data['city'] == 'Parañaque' and client_data['operation'] == 'rent':
            df_data['operation_city_1_0'] = True
        else:
            print("INVALID")
            

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


        image_file = request.files['image']

        # Check if the file is actually an image
        if image_file.filename == '':
            return "No image selected", 400
        
        image_folder = 'inputImage'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
            
        image_path = os.path.join(image_folder, image_file.filename)
        image_file.save(image_path)
        # --------------------------------------------------------------------

        score = predict_snn(client_data['lat'], client_data['lng'])

        df_data["ModelScores"] = score

        main_df = pd.DataFrame([df_data])  

        amenities = nearby_amenities(client_data['lng'], client_data['lat'], client_data['city'])

        #print(amenities)

        final_df = pd.concat([main_df, amenities], axis=1)

        client_df = pd.DataFrame([client_data])

        flood_map_5yr = gpd.read_file('Flood files/MetroManila5yr.gpkg')
        flood_map_25yr = gpd.read_file('Flood files/MetroManila25yr.gpkg')

        # Convert latitude and longitude columns in listings DataFrame to Point geometries
        listings_geometry = [Point(xy) for xy in zip(client_df['lng'], client_df['lat'])]
        listings_gdf = gpd.GeoDataFrame(client_df, geometry=listings_geometry)

        # Set CRS for amenities GeoDataFrame
        flood_map_5yr.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84 (latitude and longitude)

        # Set CRS for amenities GeoDataFrame
        flood_map_25yr.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84 (latitude and longitude)

        # Set CRS for listings GeoDataFrame
        listings_gdf.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84 (latitude and longitude)

        # Perform spatial join between property listings and flood data 5 yr
        listings_with_flood = gpd.sjoin(listings_gdf, flood_map_5yr, how="left", op="intersects")

        # Check if the spatial join was successful
        if 'Var' in listings_with_flood.columns:
            # Merge the flood threat level information back to the original DataFrame
            final_df['flood_threat_level_5_yr'] = listings_with_flood['Var']
        else:
            # Set a default value for flood threat level if not found
            final_df['flood_threat_level_5_yr'] = "Unknown"

        # Perform spatial join between property listings and flood data 25 yr
        listings_with_flood = gpd.sjoin(listings_gdf, flood_map_25yr, how="left", op="intersects")

        # Check if the spatial join was successful
        if 'Var' in listings_with_flood.columns:
            # Merge the flood threat level information back to the original DataFrame
            final_df['flood_threat_level_25_yr'] = listings_with_flood['Var']
        else:
            # Set a default value for flood threat level if not found
            final_df['flood_threat_level_25_yr'] = "Unknown"

        fault_lines = gpd.read_file("Earthquake Files/gem_active_faults_harmonized.gpkg")

        # Convert longitude and latitude columns of property listings DataFrame to Point geometries
        listings_geometry = [Point(xy) for xy in zip(client_df['lng'], client_df['lat'])]
        listings_gdf = gpd.GeoDataFrame(client_df, geometry=listings_geometry)

        # Set CRS for amenities GeoDataFrame
        fault_lines.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84 (latitude and longitude)

        # Set CRS for listings GeoDataFrame
        listings_gdf.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84 (latitude and longitude)

        # Reproject the fault_lines GeoDataFrame to EPSG:3857
        fault_lines = fault_lines.to_crs("EPSG:3857")

        # Reproject the listings GeoDataFrame to EPSG:3857
        listings_gdf = listings_gdf.to_crs("EPSG:3857")

        # Calculate minimum distance to the nearest fault line for each property listing
        final_df['min_distance_to_fault_meters'] = listings_gdf['geometry'].apply(
            lambda point: min_distance_to_fault(point, fault_lines.geometry)
        )

        columns_to_float = ["gym", "wi-fi", "swimming pool", "pay tv access", "basketball court", "jogging path",
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
            "attic", "basement", "lanai", "ducted cooling", "ducted vacuum system", "fireplace",
            "broadband internet available", "built-in wardrobes", "baths", "fully fenced",
            "air conditioning", "balcony", 'bedrooms', 'bathrooms', 'land size', 
            'floor area', 'build (year)', 'total floors', 'car spaces', 'rooms (total)', 'dryer', 'dryer.1', 'duct', 'duct.1', 'fibre', "living room"]
        
        columns_to_int = ["sqm_upper", "sqm_lower"]
        # Convert specified columns to float
        final_df[columns_to_float] = final_df[columns_to_float].astype(float)

        final_df[columns_to_int] = final_df[columns_to_int].astype(int)

        #print("Number of columns:", len(final_df.columns))

        #print("Input Data Types:")
        #for col in final_df.columns:
            #print(f"Column: {col}, Data Type: {final_df[col].dtype}")

        expected_order = ['bedrooms', 'baths', 'floor area', 'broadband internet available', 'air conditioning', 'car spaces', 'total floors', 
                          'built-in wardrobes', 'balcony', 'build (year)', 'fully fenced', 'rooms (total)', 'gym', 'wi-fi', 'swimming pool', 'pay tv access', 
                          'basketball court', 'jogging path', 'alarm system', 'lounge', 'entertainment room', 'parks', 'cctv', 'basement parking', 'elevators', 
                          'fire exits', 'function room', 'lobby', 'reception area', 'fire alarm', 'fire sprinkler system', '24-hour security', 'garden', 'secure parking', 
                          'bar', "maid's room", 'playground', 'gymnasium', 'utility room', 'billiards table', 'business center', 'club house', 'fitness center', 
                          'game room', 'meeting rooms', 'multi-purpose hall', 'shower rooms', 'sky lounge', 'smoke detector', 'social hall', 'study room', 
                          'function area', 'open space', 'gazebos', 'shops', 'study area', 'carport', 'clubhouse', 'deck', 'gazebo', 'landscaped garden', 
                          'multi-purpose lawn', 'parking lot', 'theater', 'daycare center', 'sauna', 'laundry area', 'courtyard', 'badminton court', 
                          'tennis court', 'jacuzzi', 'central air conditioning', 'health club', 'indoor spa', 'outdoor spa', 'pool bar', 'indoor pool', 
                          'drying area', 'floorboards', 'split-system heating', 'garage', 'remote garage', 'sports facilities', 'powder room', 
                          'maids room', 'library', 'spa', 'clinic', 'open car spaces', 'intercom', 'ensuite', 'pond', 'amphitheater', 'gas heating', 
                          'hydronic heating', 'indoor tree house', 'open fireplace', 'helipad', 'golf area', 'bathrooms', 'land size', 'storage room', 
                          'terrace', "driver's room", 'attic', 'basement', 'lanai', 'ducted cooling', 'ducted vacuum system', 'fireplace', 'ModelScores', 
                          'vehicle_services_nearest_distance', 'vehicle_services_walkability_score', 'vehicle_services_avg_distance', 'night_life_nearest_distance', 
                          'night_life_walkability_score', 'night_life_avg_distance', 'personal_care_nearest_distance', 'personal_care_walkability_score', 
                          'personal_care_avg_distance', 'administrative_offices_nearest_distance', 'administrative_offices_walkability_score', 
                          'administrative_offices_avg_distance', 'education_nearest_distance', 'education_walkability_score', 'education_avg_distance', 
                          'food_nearest_distance', 'food_walkability_score', 'food_avg_distance', 'general_establishments_nearest_distance', 
                          'general_establishments_walkability_score', 'general_establishments_avg_distance', 'healthcare_industries_nearest_distance', 
                          'healthcare_industries_walkability_score', 'healthcare_industries_avg_distance', 'service_providers_nearest_distance', 
                          'service_providers_walkability_score', 'service_providers_avg_distance', 'recreational_nearest_distance', 'recreational_walkability_score', 
                          'recreational_avg_distance', 'living_facilities_nearest_distance', 'living_facilities_walkability_score', 'living_facilities_avg_distance', 
                          'religion_nearest_distance', 'religion_walkability_score', 'religion_avg_distance', 'financial_nearest_distance', 'financial_walkability_score', 
                          'financial_avg_distance', 'specialized_stores_nearest_distance', 'specialized_stores_walkability_score', 'specialized_stores_avg_distance', 
                          'transportation_nearest_distance', 'transportation_walkability_score', 'transportation_avg_distance', 'flood_threat_level_5_yr', 
                          'flood_threat_level_25_yr', 'min_distance_to_fault_meters', 'vehicle_services', 'night_life', 'personal_care', 'administrative_offices', 
                          'education', 'food', 'general_establishments', 'healthcare_industries', 'service_providers', 'recreational', 'living_facilities', 
                          'religion', 'financial', 'specialized_stores', 'transportation', 'sqm_lower', 'sqm_upper', 'classification_Brand New', 'classification_Resale', 
                          'fully furnished_No', 'fully furnished_Semi', 'fully furnished_Yes', 
                          'type_encoded', 'operation_city_0_0', 'operation_city_0_1', 'operation_city_1_0', 'operation_city_1_1']

        final_df = final_df[expected_order]

        # List of columns that need to be converted to float
        columns_to_convert = [
            'vehicle_services_nearest_distance', 'vehicle_services_walkability_score', 'vehicle_services_avg_distance', 
            'night_life_nearest_distance', 'night_life_walkability_score', 'night_life_avg_distance', 
            'personal_care_nearest_distance', 'personal_care_walkability_score', 'personal_care_avg_distance', 
            'administrative_offices_nearest_distance', 'administrative_offices_walkability_score', 'administrative_offices_avg_distance', 
            'education_nearest_distance', 'education_walkability_score', 'education_avg_distance', 
            'food_nearest_distance', 'food_walkability_score', 'food_avg_distance', 
            'general_establishments_nearest_distance', 'general_establishments_walkability_score', 'general_establishments_avg_distance', 
            'healthcare_industries_nearest_distance', 'healthcare_industries_walkability_score', 'healthcare_industries_avg_distance', 
            'service_providers_nearest_distance', 'service_providers_walkability_score', 'service_providers_avg_distance', 
            'recreational_nearest_distance', 'recreational_walkability_score', 'recreational_avg_distance', 
            'living_facilities_nearest_distance', 'living_facilities_walkability_score', 'living_facilities_avg_distance', 
            'religion_nearest_distance', 'religion_walkability_score', 'religion_avg_distance', 
            'financial_nearest_distance', 'financial_walkability_score', 'financial_avg_distance', 
            'specialized_stores_nearest_distance', 'specialized_stores_walkability_score', 'specialized_stores_avg_distance', 
            'transportation_nearest_distance', 'transportation_walkability_score', 'transportation_avg_distance', 
            'vehicle_services', 'night_life', 'personal_care', 'administrative_offices', 'education', 
            'food', 'general_establishments', 'healthcare_industries', 'service_providers', 'recreational', 
            'living_facilities', 'religion', 'financial', 'specialized_stores', 'transportation'
        ]

        # Convert specified columns to float
        for column in columns_to_convert:
            final_df[column] = pd.to_numeric(final_df[column], errors='coerce')

        print(final_df['religion'])

        columns_to_fill = [
            'vehicle_services_walkability_score', 'night_life_walkability_score', 'personal_care_walkability_score',
            'administrative_offices_walkability_score', 'education_walkability_score', 'food_walkability_score',
            'general_establishments_walkability_score', 'healthcare_industries_walkability_score', 'service_providers_walkability_score',
            'recreational_walkability_score', 'living_facilities_walkability_score', 'religion_walkability_score',
            'financial_walkability_score', 'specialized_stores_walkability_score', 'transportation_walkability_score'
        ]
        
        # Fill NaN values with 0
        final_df[columns_to_fill] = final_df[columns_to_fill].fillna(0)

        vit_columns_to_fill = [
            'vehicle_services', 'night_life', 'personal_care', 'administrative_offices',
            'education', 'food', 'general_establishments', 'healthcare_industries',
            'service_providers', 'recreational', 'living_facilities', 'religion',
            'financial', 'specialized_stores', 'transportation'
        ]

        # List of columns to normalize
        columns_to_normalize = [
            'vehicle_services_nearest_distance', 'vehicle_services_avg_distance',
            'night_life_nearest_distance', 'night_life_avg_distance',
            'personal_care_nearest_distance', 'personal_care_avg_distance',
            'administrative_offices_nearest_distance', 'administrative_offices_avg_distance',
            'education_nearest_distance', 'education_avg_distance',
            'food_nearest_distance', 'food_avg_distance',
            'general_establishments_nearest_distance', 'general_establishments_avg_distance',
            'healthcare_industries_nearest_distance', 'healthcare_industries_avg_distance',
            'service_providers_nearest_distance', 'service_providers_avg_distance',
            'recreational_nearest_distance', 'recreational_avg_distance',
            'living_facilities_nearest_distance', 'living_facilities_avg_distance',
            'religion_nearest_distance', 'religion_avg_distance',
            'financial_nearest_distance', 'financial_avg_distance',
            'specialized_stores_nearest_distance', 'specialized_stores_avg_distance',
            'transportation_nearest_distance', 'transportation_avg_distance',
            'min_distance_to_fault_meters'
        ]

        # Loop through each column and fill missing values with the mode of that column
        for column in vit_columns_to_fill:
            mode_value = 0  # Get the first mode value
            final_df[column].fillna(mode_value, inplace=True)

        final_df[columns_to_normalize] = final_df[columns_to_normalize].fillna(999999)
        
        data_to_normalize = final_df[columns_to_normalize]

        normalized_data = amenity_scaler.transform(data_to_normalize)

        final_df[columns_to_normalize] = normalized_data

        flood_columns = [
            'flood_threat_level_5_yr', 'flood_threat_level_25_yr'
        ]
        
        final_df[flood_columns] = final_df[flood_columns].fillna(999999)
        
        data_to_normalize = final_df[flood_columns]

        normalized_data = flood_scaler.transform(data_to_normalize)

        final_df[flood_columns] = normalized_data

        for column in final_df.columns:
            print(f"Column: {column}, Value: {final_df[column].values[0]}")

        
        try:
            # Verify if the DataFrame is in the expected format
            # Get expected features from the model
            expected_features = get_expected_features(model)

            if not expected_features:
                return jsonify({'error': 'Failed to retrieve expected features from model'}), 500
            
            # # Check for any extra or missing features
            input_features = final_df.columns.tolist()
            missing_features = [f for f in expected_features if f not in input_features]
            extra_features = [f for f in input_features if f not in expected_features]
            
            if missing_features:
                return jsonify({'error': f'Missing features: {missing_features}'}), 400
            
            if extra_features:
                return jsonify({'error': f'Extra features: {extra_features}'}), 400
            
            # Print to check feature names
            # expected_features = get_expected_features(model)
            # print("Expected Features:", sorted(expected_features))
            # print("Input Data Columns:", sorted(final_df.columns.tolist()))
            print("Before Prediction")
            # Make prediction
            prediction_log = model.predict(final_df)

            prediction = np.exp(prediction_log)

            print("Prediction: ", prediction)
            
            # Convert prediction to JSON serializable format
            # json_prediction = json.dumps({"prediction": float(prediction)})

            print("After Prediction")

            importance = model.feature_importances_
            # Create a DataFrame for better visualization
            importance_df = pd.DataFrame({'Feature': final_df.columns, 'Importance': importance})
            sorted_importance = importance_df.sort_values(by='Importance', ascending=False)
            print(sorted_importance)

            # Select top 6 features
            top_6_importance = sorted_importance.iloc[5:11]

            # Convert to list of dictionaries
            importance_list = top_6_importance.to_dict(orient='records')

            # Use SHAP to explain the prediction
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(final_df)
            
            # Extract the SHAP values for the first instance
            shap_values_instance = shap_values[0]
                
            # Create a DataFrame for SHAP values
            shap_df = pd.DataFrame({
                'feature': final_df.columns,
                'shap_value': shap_values_instance
            })
            
            # Columns to disregard
            disregard_columns = ['operation_city_0_1', 'operation_city_0_0', 'operation_city_1_1', 'operation_city_1_0', 'type_encoded']

            # Filter out disregarded columns
            shap_df_filtered = shap_df[~shap_df['feature'].isin(disregard_columns)]

            # Sort SHAP values
            shap_df_filtered = shap_df_filtered.sort_values(by='shap_value', ascending=False)

            # Get top 3 positive and negative features
            top_positive_features = shap_df_filtered.head(3).to_dict(orient='records')
            top_negative_features = shap_df_filtered.tail(3).to_dict(orient='records')

            return jsonify({"prediction": float(prediction), "safetyScore": float(final_df['ModelScores'].values[0]), "featureImportance": importance_list, "top_positive": top_positive_features, "top_negative": top_negative_features})  # can change to just a single float value
        
        except Exception as e:
            return str(e)
    
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
    
    return safety_score

if __name__ == '__main__':
    app.run(debug=True)
