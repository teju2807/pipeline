import pymysql
import pandas as pd
import numpy as np
import scipy.stats as stats
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ===================================
# 1. CONFIGURATION AND DATA LOADING
# ===================================

# def load_config():
#     """Load database configuration from YAML file"""
#     dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#     path = os.path.join(dir_path, "config", 'config.yaml')
#     with open(path, 'r') as config_file:
#         return yaml.safe_load(config_file)
    
# def load_data(config):
#     """Load data from MySQL database"""
#     query = """
#         SELECT source_lat, source_long, cust_lat, cust_long, leg_start_time, truck_type,
#                distance, avg_speed, duration_sec 
#         FROM model_data;
#     """
    
#     try:
#         with pymysql.connect(host=config['host'], user=config['user'], 
#                              password=config['password'], database=config['database_2']) as conn:
#             df = pd.read_sql(query, conn)
#         print("✅ Data Fetched Successfully!")
#         return df
#     except Exception as e:
#         print(f"❌ Error fetching data: {e}")
#         return None

# ===================================
# 2. STATISTICAL TESTING
# ===================================


def perform_hypothesis_tests(df):
    """Perform various statistical tests on the data"""
    
    logging.info("Converting leg_start_time to datetime and extracting date parts...")
    df['leg_start_time'] = pd.to_datetime(df['leg_start_time'])
    df['year'] = df['leg_start_time'].dt.year
    df['month'] = df['leg_start_time'].dt.month
    df['day'] = df['leg_start_time'].dt.day
    df['hour'] = df['leg_start_time'].dt.hour
    
    # Feature engineering
    df['duration_min'] = df['duration_sec'] / 60
    df['speed_per_distance'] = df['avg_speed'] / df['distance']
    df['time_of_day'] = pd.cut(df['hour'], 
                               bins=[0, 6, 12, 18, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    distance_input_features = ['source_lat', 'source_long', 'cust_lat', 'cust_long', 'day', 'month', 'hour', 'truck_type']
    duration_input_features = ['source_lat', 'source_long', 'cust_lat', 'cust_long', 'day', 'hour', 'truck_type', 'avg_speed', 'distance']
    avgspeed_input_features = ['source_lat', 'source_long', 'cust_lat', 'cust_long', 'day', 'hour', 'truck_type']
    
    logging.info("Performing hypothesis tests...")
    test_results = {}
    
    # Correlation between avg_speed and duration
    corr_speed_duration = stats.pearsonr(df['avg_speed'], df['duration_sec'])
    logging.info(f"Pearson Correlation between avg_speed and duration: {corr_speed_duration[0]:.2f}, P-value: {corr_speed_duration[1]:.4f}")
    test_results['corr_speed_duration'] = {
        'statistic': corr_speed_duration[0],
        'p_value': corr_speed_duration[1],
        'significant': corr_speed_duration[1] < 0.05
    }
    
    # Correlation between distance and duration
    corr_distance_duration = stats.pearsonr(df['distance'], df['duration_sec'])
    logging.info(f"Pearson Correlation between distance and duration: {corr_distance_duration[0]:.2f}, P-value: {corr_distance_duration[1]:.4f}")
    test_results['corr_distance_duration'] = {
        'statistic': corr_distance_duration[0],
        'p_value': corr_distance_duration[1],
        'significant': corr_distance_duration[1] < 0.05
    }
    
    # T-Test: Short vs Long Trips
    median_distance = df['distance'].median()
    short_trips = df[df['distance'] <= median_distance]['avg_speed']
    long_trips = df[df['distance'] > median_distance]['avg_speed']
    
    t_stat, p_value = stats.ttest_ind(short_trips, long_trips, equal_var=False)
    logging.info(f"T-Test for avg_speed between short and long trips: T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    test_results['ttest_short_long'] = {
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Correlation Tests for Various Features
    for feature in distance_input_features:
        stat, p_value = stats.shapiro(df[feature])
        method = "Pearson" if p_value > 0.05 else "Spearman"
        corr_func = stats.pearsonr if method == "Pearson" else stats.spearmanr
        corr, p = corr_func(df[feature], df['distance'])
        
        test_results.setdefault('distance', {})[feature] = {
            'method': method,
            'correlation': corr,
            'p_value': p,
            'significant': p < 0.05,
            'compared_with': 'distance'
        }
        logging.debug(f"{method} Correlation between {feature} and distance: {corr:.2f}, P-value: {p:.4f}")
    
    for feature in duration_input_features:
        stat, p_value = stats.shapiro(df[feature])
        method = "Pearson" if p_value > 0.05 else "Spearman"
        corr_func = stats.pearsonr if method == "Pearson" else stats.spearmanr
        corr, p = corr_func(df[feature], df['duration_sec'])
        
        test_results.setdefault('duration_sec', {})[feature] = {
            'method': method,
            'correlation': corr,
            'p_value': p,
            'significant': p < 0.05,
            'compared_with': 'duration_sec'
        }
        logging.debug(f"{method} Correlation between {feature} and duration_sec: {corr:.2f}, P-value: {p:.4f}")
    
    for feature in avgspeed_input_features:
        stat, p_value = stats.shapiro(df[feature])
        method = "Pearson" if p_value > 0.05 else "Spearman"
        corr_func = stats.pearsonr if method == "Pearson" else stats.spearmanr
        corr, p = corr_func(df[feature], df['avg_speed'])
        
        test_results.setdefault('avg_speed', {})[feature] = {
            'method': method,
            'correlation': corr,
            'p_value': p,
            'significant': p < 0.05,
            'compared_with': 'avg_speed'
        }
        logging.debug(f"{method} Correlation between {feature} and avg_speed: {corr:.2f}, P-value: {p:.4f}")
    
    return test_results

# config = load_config()
# df = load_data(config)
# if len(df) >= 40000:
#     print(perform_hypothesis_tests(df))
# else :
#     print("data is not sufficeient for further use")
