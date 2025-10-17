import pandas as pd
import requests
import concurrent.futures
from sqlalchemy import create_engine,text
import os
import yaml
import numpy as np
from urllib.parse import quote
import pymysql
import logging
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Set up directory paths
dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load config
path = os.path.join(dir_path, "config", 'config.yaml')
with open(path, 'r') as config_file:
    DB_CONFIG = yaml.safe_load(config_file)

def run_query(query, data):
    """Executes SQL queries via the API with optional parameters."""
    logging.info(f"Executing query: {query}")
    # logging.info(f"dataaaaaaaaa : {data}")
    
    payload = {
        "query": query,
        "data": data if data else None  # ✅ Ensure an empty list is sent if there's no data
    }

    try:
        response = requests.post(DB_CONFIG['connection_url'], json=payload)
        response.raise_for_status()
        data = response.json()
        # logging.info(f"API Response: {data}")

        if isinstance(data, list):  
            return pd.DataFrame(data)  # Convert to DataFrame if response is tabular
        # ✅ Check if response is a dictionary and contains an 'error' key
        elif isinstance(data, dict):
            if "error" in data:
                logging.error(f"API Error: {data['error']}")
                return pd.DataFrame()  # Return empty DataFrame on error
            return data  # Return dictionary if no error
        return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        logging.error(f"API Request Error: {e}")
        return pd.DataFrame()


def get_last_processed_id(database2,table_name3):
    logging.info("get_last_processed_id")
    """Retrieve last processed ID from MySQL or a local file."""
    query = f"SELECT MAX(id) FROM {database2}.{table_name3};"  # Create a table to track processed ID
    try:
        result = run_query(query,data=None)
        logging.info(f"resulttt {result} ")
        # return int(result.iloc[0,0]) if result else 0
        # Explicit check for DataFrame and value existence
        if (
            isinstance(result, pd.DataFrame)
            and not result.empty
            and result.iloc[0, 0] is not None
        ):
            return int(result.iloc[0, 0])
        else:
            return 0
    except Exception as e:
        logging.info(f"Error fetching last processed ID: {e}")
        return 0

def update_last_processed_id(new_max_id,database2,table_name3):
    """Update the last processed ID after completion. If the table does not exist, create it first."""
    
    new_max_id = int(new_max_id)
    
    create_db_query = f"CREATE DATABASE IF NOT EXISTS {database2}"
    db_response = run_query(create_db_query,data=None)
        
    # # Check if the response contains the success message
    # if not db_response.empty and "Query executed successfully" in db_response.iloc[0, 0]:
    logging.info(f"Database '{database2}' created or already exists : {db_response}")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{database2}`.`{table_name3}` (
        id BIGINT PRIMARY KEY
    );
    """
    
    insert_query = f"""
    INSERT INTO `{database2}`.`{table_name3}` (id) 
    VALUES (%s) 
    ON DUPLICATE KEY UPDATE id = VALUES(id);
    """
    
    run_query(create_table_query, data=None)
    run_query(insert_query, new_max_id)
    logging.info(f"Last processed ID updated to {new_max_id}")

def fetch_and_filter_data(database1,table_name1,database2,table_name3):
    """
    Fetches data from MySQL and filters based on conditions.
    """
    last_max_id = get_last_processed_id(database2,table_name3)
    logging.info(f"last_max_id : {last_max_id}")
    query = f"SELECT * FROM {database1}.{table_name1} WHERE id > {last_max_id};"
    
    try:
        
        df = run_query(query,data=None)
        logging.info(f"data : {len(df)}")
        
        if df.empty:
            logging.info("No new data to process.")
            return None
        
        logging.info(f"Processing {len(df)} new records from ID {last_max_id + 1} to {df['id'].max()}.")

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

    except requests.exceptions.RequestException as e:
        logging.info(f"Error fetching data: {e}")
        return None, last_max_id

@lru_cache(maxsize=500)  # Cache the results for already computed distances
def fetch_route_cached(start_lng, start_lat, end_lng, end_lat):
    """Fetches route from OSRM API with caching"""
    try:
        url = f"https://routing.openstreetmap.de/routed-car/route/v1/driving-hgv/{start_lng},{start_lat};{end_lng},{end_lat}?overview=false"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['routes'][0]['distance'] if 'routes' in data and data['routes'] else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Route Fetch Error: {e}")
        return None

def process_row(index, row):
    """Processes a single row to calculate distance"""
    distance = fetch_route_cached(row['source_long'], row['source_lat'], row['cust_long'], row['cust_lat'])
    return index, distance

def calculate_distances(df):
    """Optimized parallel API calls with limited threads"""
    df['hgv_calculated_distance'] = None  # Initialize column

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, index, row): index for index, row in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            index, distance = future.result()
            if distance is not None:
                df.at[index, 'hgv_calculated_distance'] = distance
                logging.info(f"Row {index}: Distance calculated: {distance} meters.")
    
    return df



def merge_with_original(filtered_data, result_df):
    """Merges calculated distance results with the original filtered data."""
    final_df = filtered_data.merge(result_df, on=['source_lat', 'source_long', 'cust_lat', 'cust_long'], how='left')
    return final_df



def store_to_db(df, database2, table_name2, batch_size=1000):
    """Stores the DataFrame into MySQL using an API-based approach."""
    try:
        # ✅ Create Database if not exists
        create_db_query = f"CREATE DATABASE IF NOT EXISTS {database2}"
        run_query(create_db_query,data=None)
        logging.info(f"Database '{database2}' checked/created.")

        # ✅ Create Table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{database2}`.`{table_name2}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            trip_id BIGINT,
            leg_no INT,
            plant_lat DOUBLE,
            plant_long DOUBLE,
            cust_lat DOUBLE,
            cust_long DOUBLE,
            leg_start_time DATETIME,
            leg_end_time DATETIME,
            distance INT,
            truck_type INT,
            duration BIGINT,
            avg_speed DOUBLE,
            leg_hit_status VARCHAR(50),
            probable_lat DOUBLE,
            probable_long DOUBLE,
            gps_data_file_path VARCHAR(255),
            source_id INT,
            destination_id INT,
            plant_id INT,
            hierarchy_path VARCHAR(255),
            source_lat DOUBLE,
            source_long DOUBLE,
            duration_sec FLOAT,
            count INT,
            hgv_calculated_distance FLOAT,
            dist_diff VARCHAR(10)
        );
        """
        run_query(create_table_query,data=None)

        if df.empty:
            logging.warning("No data to insert. Exiting insert process.")
            return
        df.replace({np.nan: None}, inplace=True)  # Replace NaN with None

        insert_query = f"""
            INSERT INTO `{database2}`.`{table_name2}` 
            (trip_id, leg_no, plant_lat, plant_long, cust_lat, cust_long, 
            leg_start_time, leg_end_time, distance, truck_type, duration, avg_speed, 
            leg_hit_status, probable_lat, probable_long, gps_data_file_path, 
            source_id, destination_id, plant_id, hierarchy_path, source_lat, source_long,
            duration_sec, count, hgv_calculated_distance, dist_diff)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data_dicts = df.to_dict(orient="records")
        data_list = [
            (
                d["trip_id"], d["leg_no"], d["plant_lat"], d["plant_long"], d["cust_lat"], d["cust_long"],
                d["leg_start_time"], d["leg_end_time"], d["distance"], d["truck_type"], d["duration"], d["avg_speed"],
                d["leg_hit_status"], d["probable_lat"], d["probable_long"], d["gps_data_file_path"],
                d["source_id"], d["destination_id"], d["plant_id"], d["hierarchy_path"], d["source_lat"], d["source_long"],
                d["duration_sec"], d["count"], d["hgv_calculated_distance"], d["dist_diff"]
            )
            for d in data_dicts
        ]

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            try:
                run_query(insert_query, batch)  # Send batch to API
                logging.info(f"Inserted batch {i // batch_size + 1}/{len(data_list) // batch_size + 1}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error inserting batch: {e}", exc_info=True)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error inserting data: {e}", exc_info=True)
        
def create_indexes(database2, table_name2):
    """Creates necessary indexes on the table after checking if they exist."""

    index_columns = {
        "idx_trip_id": ["trip_id"],
        "idx_source_lat_long": ["source_lat", "source_long"],
        "idx_cust_lat_long": ["cust_lat", "cust_long"],
        "idx_leg_hit_status": ["leg_hit_status"],
        "idx_distance": ["distance"],
        "idx_duration": ["duration"],
        "idx_avg_speed": ["avg_speed"],
    }

    for index_name, columns in index_columns.items():
        # Prepare column string for index creation
        column_str = ", ".join(columns)
        
        # Query to check if index exists
        check_query = f"""
            SELECT COUNT(1) as index_exists
            FROM information_schema.STATISTICS
            WHERE table_schema = '{database2}'
              AND table_name = '{table_name2}'
              AND index_name = '{index_name}';
        """

        try:
            result = run_query(check_query, data=None)
            index_exists = result.iloc[0]['index_exists'] if not result.empty else 0

            if index_exists == 0:
                create_index_query = f"CREATE INDEX {index_name} ON {database2}.{table_name2} ({column_str});"
                run_query(create_index_query, data=None)
                logging.info(f"Created index: {index_name} on ({column_str})")
            else:
                logging.info(f"Index already exists: {index_name}")
        except Exception as e:
            logging.warning(f"Failed to check or create index {index_name}: {e}")
        
            
     
def main():
    database1 = 'eta'
    table_name1 = 'eta_data'
    database2 = 'model_data_db1'
    table_name2 = 'model_data1'
    table_name3 = 'processed_data1'
    filtered_data, new_max_id = fetch_and_filter_data(database1,table_name1,database2,table_name3)
    if filtered_data is None or filtered_data.empty:
        logging.info("No valid data found.")
        return

    grouped_data = (
    filtered_data
    .groupby(['source_lat', 'source_long', 'cust_lat', 'cust_long'])
    .size()
    .reset_index(name='count')
    )

    logging.info("Processing route distances...")
    logging.info(f"length of grouped data ,{len(grouped_data)}")
    logging.info(f"1111111111111111 {grouped_data}")
    result_df = calculate_distances(grouped_data)
    logging.info(f"length of result data {len(result_df)}")
    logging.info(f"result_df {result_df}")

    final_df = merge_with_original(filtered_data, result_df)
    data = final_df[final_df['hgv_calculated_distance'] != 0]

    data['dist_diff'] = np.where(
        abs(data['distance'] - data['hgv_calculated_distance']) / data['hgv_calculated_distance'] <= 0.3,
        "TRUE",
        "FALSE"
    )

    latest_df = data[(data['dist_diff'] == "TRUE") & (data['avg_speed'] >= 0.3)]
    logging.info(f'latest_dffffffffffffffff {len(latest_df)}')
    
    logging.info("Storing processed rows into DB.")
    store_to_db(latest_df,database2,table_name2, batch_size=1000)
    
    create_indexes(database2, table_name2)

    # Update last processed ID
    update_last_processed_id(new_max_id,database2,table_name3)

if __name__ == "__main__":
    main()
