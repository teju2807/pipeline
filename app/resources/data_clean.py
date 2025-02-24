import pandas as pd
import requests
import logging
import concurrent.futures
from sqlalchemy import create_engine
import os
import yaml
import numpy as np

# Set up logging
logging.basicConfig(filename='eta.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

 # Set up directory paths
dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load config
path = os.path.join(dir_path,"config", 'config.yaml')
with open(path, 'r') as config_file:
    DB_CONFIG = yaml.safe_load(config_file)

def get_db_engine():
    """Returns a SQLAlchemy engine for MySQL."""
    return create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def fetch_and_filter_data():
    """
    Fetches data from MySQL and filters based on conditions.
    
    Returns:
        DataFrame: Filtered data
    """
    query = "SELECT * FROM eta.eta_data;"
    
    try:
        engine = get_db_engine()
        df = pd.read_sql(query, engine)
        logging.info("Data loaded successfully from MySQL.")

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

        return df_latest

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def fetch_route(start, end):
    """
    Fetches a route from the OSRM API between two points.

    Args:
        start (dict): {'lat': value, 'lng': value}
        end (dict): {'lat': value, 'lng': value}

    Returns:
        float: Distance in meters if successful, None otherwise.
    """
    try:
        url = f"https://router.project-osrm.org/route/v1/driving-hgv/{start['lng']},{start['lat']};{end['lng']},{end['lat']}?overview=full&geometries=geojson"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('routes') and len(data['routes']) > 0:
            return data['routes'][0]['distance']
        else:
            logging.warning(f"No route found between {start} and {end}.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Route Fetch Error: {e}")
        return None

def process_row(index, row):
    """
    Processes a single row to calculate distance.

    Args:
        index (int): Row index.
        row (pd.Series): Row data.

    Returns:
        tuple: (index, calculated_distance)
    """
    start_point = {'lat': row['source_lat'], 'lng': row['source_lon']}
    end_point = {'lat': row['cust_lat'], 'lng': row['cust_long']}

    logging.info(f"Fetching route for row {index}: {start_point} -> {end_point}")
    
    try:
        distance = fetch_route(start_point, end_point)
        return (index, distance)
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")
        return (index, None)

def calculate_distances(df):
    """
    Uses multithreading to calculate distances for all rows.

    Args:
        df (pd.DataFrame): Grouped DataFrame.

    Returns:
        pd.DataFrame: DataFrame with calculated distances.
    """
    df['hgv_calculated_distance'] = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_row, index, row): index for index, row in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            index, distance = future.result()
            if distance is not None:
                df.at[index, 'hgv_calculated_distance'] = distance
                logging.info(f"Row {index}: Distance calculated: {distance} meters.")
    
    return df

def merge_with_original(filtered_data, result_df):
    """
    Merges calculated distance results with the original filtered data.

    Args:
        filtered_data (pd.DataFrame): Original filtered data.
        result_df (pd.DataFrame): DataFrame with calculated distances.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    final_df = filtered_data.merge(result_df, on=['source_lat', 'source_lon', 'cust_lat', 'cust_long'], how='left')
    return final_df

def store_to_db(df):
    """
    Stores the merged DataFrame into MySQL.
    """
    table_name = 'model_data'
    database = 'model_data_db'

    try:
        # Connect to MySQL server without specifying a database
        temp_engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}")

        # Create database if not exists
        with temp_engine.connect() as conn:
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {database};")
            logging.info(f"Database '{database}' created or already exists.")

        # Now, connect to the new database
        engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        # return engine
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        logging.info(f"Data successfully stored in '{table_name}' table."
                     )
    except Exception as e:
        logging.error(f"Error storing data: {e}")


def main():
    
    # Step 1: Fetch and filter data
    filtered_data = fetch_and_filter_data()
    if filtered_data is None or filtered_data.empty:
        logging.warning("No valid data found.")
        return

    # Step 2: Group by unique locations
    grouped_data = filtered_data[['source_lat', 'source_lon', 'cust_lat', 'cust_long']].drop_duplicates()

    # Step 3: Calculate distances
    result_df = calculate_distances(grouped_data)

    # Step 4: Merge results with original data
    final_df = merge_with_original(filtered_data, result_df)

    # Step 5: Filter out zero distances
    data = final_df[final_df['hgv_calculated_distance'] != 0]

    # Step 6: Compute distance difference flag
    data['dist_diff'] = np.where(
        abs(data['distance'] - data['hgv_calculated_distance']) / data['hgv_calculated_distance'] <= 0.3,
        "TRUE",
        "FALSE"
    )

    # Step 7: Filter for distances with more than 30% difference
    latest_df = data[(data['dist_diff'] == "TRUE") & (data['avg_speed'] >= 0.3)]

    # Step 5: Store in MySQL
    store_to_db(latest_df)

# Run the process
if __name__ == "__main__":
    main()
