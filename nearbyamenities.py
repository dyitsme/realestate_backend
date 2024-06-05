from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd
import math
import json
from shapely.geometry import Point
from geopy.distance import geodesic

app = Flask(__name__)
CORS(app)

def nearby_amenities(longitude, latitude, city):
    # Retrieve the JSON string from form data
    json_str = request.form.get('coords')

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_str)

    # Access latitude and longitude values from the dictionary
    latitude = data['lat']
    longitude = data['lng']

    print(latitude)
    print(longitude)

    point = Point(longitude, latitude)
    print(Point)
    # Your logic to compute nearby amenities
    amenities = compute_nearby_amenities(point, city)

    return amenities

def read_amenities_file(city):
    # Load GeoJSON file of amenities into a GeoDataFrame
    if city == 'pasig':
        amenities_gdf = gpd.read_file('Amenity Files/pasig_amenities.geojson')
    elif city == 'paranaque':
        amenities_gdf = gpd.read_file('Amenity Files/pq_amenities.geojson')

    amenities_gdf_projected = amenities_gdf

    # Specify the distance threshold (in meters)
    distance_threshold = 1000  # Adjust as needed

def category_mapping():
    category_mapping = {
        'vehicle_services': {'amenities': ['fuel', 'car_wash', 'compressed_air', 'car_repair', 'car', 'car_parts', 'gas', 'battery'], 'tags': ['shop', 'amenity', 'car_parts']},
        
        'night_life': {'amenities': ['pub', 'bar', 'nightclub', ], 'tags': ['amenity', 'shop']},
        
        'personal_care': {'amenities': ['massage_chair', 'cosmetics', 'spa', 'hairdresser', 'boutique', 'optician', 'nails', 'cosmetics'], 
                        'tags': ['shop', 'amenity', 'beauty']},
        
        'administrative_offices': {'amenities': ['townhall', 'police', 'fire_station', 'courthouse', 'community_centre', 'post_office', 'government', 'administrative'
                                                ], 'tags': ['amenity', 'shop', 'office']},
        
        'education': {'amenities': ['school', 'college', 'kindergarten', 'prep_school', 'driving_school', 'special_needs'], 
                    'tags': ['amenity', 'shop', 'specialized_education', 'government']},
        
        'food': {'amenities': ['restaurant', 'fast_food', 'cafe', 'bbq', 'food_court', 'vending_machine', 'ice_cream', 'drinking_water', 'bakery', 
                            'pastry', 'yes', 'only'], 'tags': ['amenity', 'shop', 'delivery', 'drive_through', 'takeaway']},
        
        'general_establishments': {'amenities': ['toilets', 'public_building', 'post_box', 'waste_basket', 'mall', 'kiosk', 'vacant', 'yes',
                                                'general'], 'tags': ['amenity', 'shop']},
        
        'healthcare_industries': {'amenities': ['pharmacy', 'hospital', 'veterinary', 'doctors', 'clinic', 'dentist', 'health_post', 'childcare', 'blood_check', 'pedia', 
                                                'community_health_worker'], 'tags': ['amenity', 'shop', 'healthcare:speciality', 'healthcare']},
        
        'service_providers': {'amenities': ['car_sharing', 'internet_cafe', 'telephone', 'post_depot', 'car_rental', 'payment_terminal', 'loading_dock', 
                                            'payment_centre', 'crematorium', 'events_venue', 'shelf', 'massage', 'laundry', 'butcher', 'ticket', 'travel_agency',
                                        'tailor', 'trade', 'printing', 'copyshop', 'pet_grooming', 'funeral_directors', 'telecommunication', 'wlan', 'terminal', 
                                        'service', 'yes'], 'tags': ['amenity', 'shop', 'internet_access']},
        
        'recreational': {'amenities': ['bench', 'cinema', 'fountain', 'training', 'smoking_area', 'casino', 'library'], 'tags': ['amenity', 'shop']},
        
        'living_facilities': {'amenities': ['shelter', 'lounge'], 'tags': ['amenity', 'shop']},
        
        'religion': {'amenities': ['place_of_worship', 'monastery'], 'tags': ['amenity', 'shop']},
        
        'financial': {'amenities': ['bank', 'bureau_de_change', 'atm', 'money_transfer', 'money_lender'], 'tags': ['amenity', 'shop']},
        
        'specialized_stores': {'amenities': ['convenience', 'marketplace', 'public_market', 'supermarket', 'books', 'electronics', 'fabric', 
                                            'art', 'garden_centre', 'computer', 'health', 'water', 'baby_goods', 'furniture', 'water_station', 
                                            'doityourself', 'hardware', 'advertising', 'guns', 'pet', 'motorcycle', 'department_store', 'bicycle',
                                            'seafood', 'greengrocer', 'tyres', 'pawnbroker', 'video_games', 'hifi', 'shoes', 'tea', 'clothes',
                                            'stationery', 'wine;alcohol', 'beauty', 'appliance', 'e-cigarette', 'confectionery', 'alcohol', 'sports',
                                            'beverages', 'variety_store', 'coffee', 'medical_supply', 'chemist', 'party', 'health_food',
                                            'paint', 'gift', 'jewelry', 'bag', 'perfumery', 'bed', 'watches', 'rice', 'electrical', 'florist', 'photo',
                                            'houseware', 'fashion_accessories', 'frozen_food', 'interior_decoration', 'model', 'farm'], 'tags': ['amenity', 'shop']},
        
        'transportation': {'amenities': ['taxi', 'parking', 'parking_entrance', 'bus_station', 'motorcycle_parking', 'bicycle_parking', 
                                        'parking_space', 'subway_entrance'], 'tags': ['amenity', 'shop', 'railway']},
        # Add more categories and corresponding amenities/tags as needed
    }
    

def walkability_score(distance):
    
    if(distance > 1000):
        return 0
    
    e = math.e
    
    walk_score = e**(-5*(distance/1000)**5)
    return walk_score

def compute_nearby_amenities(point, city):
    read_amenities_file(city)
    category_mapping()
    walkability_score()

    amenities_data = pd.DataFrame()

    # Iterate over each listing
    listing_shortest_distances = {category: float('inf') for category in category_mapping}
    listing_distance_sum = {category: 0 for category in category_mapping}
    listing_num = {category: 0 for category in category_mapping}
        
        for amenity_idx, amenity_row in amenities_gdf_projected.iterrows():
            # print("LISTING",listing.geometry)
            # print("AMENITY",amenity_row.geometry)
            # print(amenity_row.amenity)
            
            # amen_distance = get_route_coordinates((listing.geometry.y,listing.geometry.x),(amenity_row.geometry.y,amenity_row.geometry.x))
            # print(amen_distance)   
            
            distance = geodesic((point.geometry.y, point.geometry.x), (amenity_row.geometry.y, amenity_row.geometry.x)).meters
            # print("Distance using geodesic:", distance_geodesic)
            

            # Check if the amenity falls under any category
            for category, details in category_mapping.items():
                
                column_name1 = category + "_nearest_distance"
                column_name2 = category + "_walkability_score"
                column_name3 = category + "_avg_distance"
                column_name = category

                if amenity_row.amenity in details['amenities']:
                    if (distance <= distance_threshold):
                        
                        listing_distance_sum[category] += distance
                        listing_num[category] += 1
                        amenities_data.at[0, column_name3] = listing_distance_sum[category] / listing_num[category]
                        amenities_data.at[0, column_name] = listing_num[category]
                        
                        # Update shortest distance if the current distance is shorter
                        if (listing_shortest_distances[category] is None or distance < listing_shortest_distances[category]):
                            listing_shortest_distances[category] = distance
                            amenities_data.at[0, column_name1] = distance
                            amenities_data.at[0, column_name2] = walkability_score(distance)
                                                                
                    break  # No need to continue checking other categories for this amenity
    
        for category, distance in amenities_data.items():
            print(f"Shortest distance to {category}: {distance}")

    return amenities_data


if __name__ == '__main__':
    app.run(debug=True)
