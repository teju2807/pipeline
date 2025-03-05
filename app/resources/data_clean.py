import pandas as pd
import requests
import concurrent.futures
from sqlalchemy import create_engine
import os
import yaml
import numpy as np
from urllib.parse import quote
import pymysql

# Set up directory paths
dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load config
path = os.path.join(dir_path, "config", 'config.yaml')
with open(path, 'r') as config_file:
    DB_CONFIG = yaml.safe_load(config_file)

# DB_CONFIG = {
#     "user": os.getenv("DB_USER", "default_user"),
#     "password": os.getenv("DB_PASSWORD", "default_pass"),
#     "host": os.getenv("DB_HOST", "127.0.0.1"),
#     "database": os.getenv("DB_DATABASE", "eta"),
# }

# encoded_password = quote(DB_CONFIG['password'])

# def get_db_engine():
#     """Returns a SQLAlchemy engine for MySQL."""
#     return create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def get_last_processed_id():
    """Retrieve last processed ID from MySQL or a local file."""
    query = "SELECT MAX(id) FROM processed_data;"  # Create a table to track processed ID
    try:
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_CONFIG['database'])
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0  # Return last max ID or 0 if not found
    except Exception as e:
        print(f"Error fetching last processed ID: {e}")
        return 0

def update_last_processed_id(new_max_id):
    """Update the last processed ID after completion."""
    query = f"INSERT INTO processed_data (id) VALUES ({new_max_id}) ON DUPLICATE KEY UPDATE id = {new_max_id};"
    try:
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_CONFIG['database'])
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()
    except Exception as e:
        print(f"Error updating last processed ID: {e}")

def fetch_and_filter_data():
    """
    Fetches data from MySQL and filters based on conditions.
    """
    last_max_id = get_last_processed_id()
    query = f"SELECT * FROM eta.eta_data WHERE id > {last_max_id};"
    
    try:
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_CONFIG['database'])
        print("Connected successfully")
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("No new data to process.")
            return None
        
        print(f"Processing {len(df)} new records from ID {last_max_id + 1} to {df['id'].max()}.")

        # Apply filtering conditions
        df_filter = df[(df['leg_hit_status'].isin(['GEOFENCEHIT', 'PINCODEHIT']))]

        df_updated = df_filter[
            (df_filter['source_lat'] != 0) & (df_filter['source_long'] != 0) & 
            (df_filter['cust_lat'] != 0) & (df_filter['cust_long'] != 0) &
            (df_filter['distance'] > 0) & (df_filter['duration'] > 0) & (df_filter['avg_speed'] > 0)
        ]

        df_latest = df_updated[(df_updated['source_lat'] != df_updated['cust_lat']) | (df_updated['source_long'] != df_updated['cust_long'])]

        # Convert duration from ms to seconds
        df_latest['duration_sec'] = df_latest['duration'] / 1000

        return df_latest, df['id'].max()  # Return both filtered data and max ID

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, last_max_id

def fetch_route(start, end):
    """
    Fetches a route from the OSRM API between two points.
    """
    try:
        url = f"https://router.project-osrm.org/route/v1/driving-hgv/{start['lng']},{start['lat']};{end['lng']},{end['lat']}?overview=full&geometries=geojson"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('routes') and len(data['routes']) > 0:
            return data['routes'][0]['distance']
        else:
            print(f"No route found between {start} and {end}.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Route Fetch Error: {e}")
        return None

def process_row(index, row):
    """
    Processes a single row to calculate distance.
    """
    start_point = {'lat': row['source_lat'], 'lng': row['source_long']}
    end_point = {'lat': row['cust_lat'], 'lng': row['cust_long']}

    print(f"Fetching route for row {index}: {start_point} -> {end_point}")
    
    try:
        distance = fetch_route(start_point, end_point)
        return (index, distance)
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return (index, None)

def calculate_distances(df):
    """
    Uses multithreading to calculate distances for all rows.
    """
    df['hgv_calculated_distance'] = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_row, index, row): index for index, row in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            index, distance = future.result()
            if distance is not None:
                df.at[index, 'hgv_calculated_distance'] = distance
                print(f"Row {index}: Distance calculated: {distance} meters.")
    
    return df

def merge_with_original(filtered_data, result_df):
    """Merges calculated distance results with the original filtered data."""
    final_df = filtered_data.merge(result_df, on=['source_lat', 'source_long', 'cust_lat', 'cust_long'], how='left')
    return final_df

def store_to_db(df):
    """Stores the merged DataFrame into MySQL."""
    table_name = 'model_data'
    database = 'model_data_db'

    try:
        temp_engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}")
        with temp_engine.connect() as conn:
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {database};")
            print(f"Database '{database}' created or already exists.")

        engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data successfully stored in '{table_name}' table.")
    except Exception as e:
        print(f"Error storing data: {e}")
        
        
def main():
    filtered_data, new_max_id = fetch_and_filter_data()
    if filtered_data is None or filtered_data.empty:
        print("No valid data found.")
        return

    grouped_data = filtered_data[['source_lat', 'source_long', 'cust_lat', 'cust_long']].drop_duplicates()
    print("Processing route distances...")
    result_df = calculate_distances(grouped_data)

    final_df = merge_with_original(filtered_data, result_df)
    data = final_df[final_df['hgv_calculated_distance'] != 0]

    data['dist_diff'] = np.where(
        abs(data['distance'] - data['hgv_calculated_distance']) / data['hgv_calculated_distance'] <= 0.3,
        "TRUE",
        "FALSE"
    )

    latest_df = data[(data['dist_diff'] == "TRUE") & (data['avg_speed'] >= 0.3)]
    print(f"Storing {len(latest_df)} processed rows into DB.")
    store_to_db(latest_df)

    # Update last processed ID
    update_last_processed_id(new_max_id)

if __name__ == "__main__":
    main()

