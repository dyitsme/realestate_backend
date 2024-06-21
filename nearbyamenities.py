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

    #point = Point(longitude, latitude)

    # Your logic to compute nearby amenities
    amenities = compute_nearby_amenities(latitude, longitude, city)

    return amenities

def walkability_score(distance):
    
    if(distance > 1000):
        return 0
    
    e = math.e
    
    walk_score = e**(-5*(distance/1000)**5)
    return walk_score

def compute_nearby_amenities(lat, lng, city):

    df_data = {
        'lat': lat,
        'lng': lng
    }

    df = pd.DataFrame([df_data])
    
    listings_geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]
    listings_gdf = gpd.GeoDataFrame(df, geometry=listings_geometry)

    if city == 'Pasig':
        amenities_gdf = gpd.read_file('pasig_amenities.geojson')
    elif city == 'Parañaque':
        amenities_gdf = gpd.read_file('pq_amenities.geojson')

    listings_gdf_projected = listings_gdf
    amenities_gdf_projected = amenities_gdf

    print(listings_gdf_projected.head())

    # Specify the distance threshold (in meters)
    distance_threshold = 1000  # Adjust as needed

    amenities_data = pd.DataFrame()

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

    # Assuming you have an existing GeoDataFrame named 'gdf'
    # Create an empty column for each category name
    for category_name in category_mapping:
        column_name1 = category_name + "_nearest_distance"
        column_name2 = category_name + "_walkability_score"
        column_name3 = category_name + "_avg_distance"
        listings_gdf_projected[column_name1] = None
        listings_gdf_projected[column_name2] = None
        listings_gdf_projected[column_name3] = None
        
        column_name = category_name
        listings_gdf_projected[column_name] = None
    #print("category")

    # Iterate over each listing
    for idx, listing in listings_gdf_projected.iterrows():
        listing_shortest_distances = {category: float("999999")  for category in category_mapping}
        listing_distance_sum = {category: 0 for category in category_mapping}
        listing_num = {category: 0 for category in category_mapping}
        
        print("Listing#", idx)

        for amenity_idx, amenity_row in amenities_gdf_projected.iterrows():
            # print("LISTING",listing.geometry)
            # print("AMENITY",amenity_row.geometry)
            # print(amenity_row.amenity)
            
            # amen_distance = get_route_coordinates((listing.geometry.y,listing.geometry.x),(amenity_row.geometry.y,amenity_row.geometry.x))
            # print(amen_distance)   
            

            distance = geodesic((listing.geometry.y, listing.geometry.x), (amenity_row.geometry.y, amenity_row.geometry.x)).meters
            #print("Distance using geodesic:", distance)
            

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
                        listings_gdf_projected.at[idx, column_name3] = float(listing_distance_sum[category] / listing_num[category])
                        listings_gdf_projected.at[idx, column_name] = float(listing_num[category])

                        # Update shortest distance if the current distance is shorter
                        if (listing_shortest_distances[category] is None or distance < listing_shortest_distances[category]):
                            listing_shortest_distances[category] = float(distance)
                            listings_gdf_projected.at[idx, column_name1] = float(distance)
                            listings_gdf_projected.at[idx, column_name2] = float(walkability_score(distance))
                                                                
                    break  # No need to continue checking other categories for this amenity
    
            
        print(f"Listing {idx}:")
        for category, distance in listing_shortest_distances.items():
            print(f"Shortest distance to {category}: {distance}")

    listings_gdf_projected.drop(columns=['geometry'], inplace=True)
    return listings_gdf_projected


if __name__ == '__main__':
    app.run(debug=True)
