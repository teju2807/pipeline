import optuna
from pycaret.regression import *
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge, Lasso, LinearRegression, Lars, LassoLars, ElasticNet, HuberRegressor, OrthogonalMatchingPursuit, PassiveAggressiveRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statistics_analysis import perform_hypothesis_tests
from hummingbird.ml import convert
import torch
import os
import yaml
import pandas as pd
import pymysql
import logging
import requests
import joblib


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config():
    """Load database configuration from YAML file"""
    dir_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path = os.path.join(dir_path, "config", 'config.yaml')
    with open(path, 'r') as config_file:
        return yaml.safe_load(config_file)

def load_data(config):
    """Fetch model data from MySQL database"""
    query = """
        SELECT source_lat, source_long, cust_lat, cust_long, leg_start_time, truck_type,
               distance, avg_speed, duration
        FROM demo.model_data;
    """
    try:
        df = run_query(query, config)
        logging.info("✅ Data Fetched Successfully!")
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data: {e}")
        return None

def run_query(query, config):
    """Sends a SQL query to the Flask API via HTTP POST request."""
    url = f"{config['connection_url']}"
    payload = {"query": query}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making API request: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """Preprocess dataset and split into train and test sets"""
    df['no_count'] = df.groupby(['source_lat', 'source_long', 'cust_lat', 'cust_long'])['source_lat'].transform('count')
    training_set = df[df['no_count'] == 1]
    remaining_data = df[df['no_count'] > 1]
    
    def split_group(group):
        train_count = round(0.8 * len(group))
        return group.iloc[:train_count], group.iloc[train_count:]
    
    train_groups, test_groups = zip(*[split_group(group) for _, group in remaining_data.groupby(['source_lat', 'source_long', 'cust_lat', 'cust_long'])])
    train_set = pd.concat([training_set] + list(train_groups)).reset_index(drop=True)
    test_set = pd.concat(test_groups).reset_index(drop=True)
    
    logging.info("✅ Data Preprocessing Completed")
    return train_set, test_set


def train_pycaret_model(train_set, test_set, features, target, model_name, model_dir):
    """Train a regression model using PyCaret and return the best model"""
    if not features:
        logging.warning(f"⚠️ No significant features found for {model_name}. Skipping...")
        return None, None

    logging.info(f"🔧 Setting up PyCaret for {model_name}...")
    setup(data=pd.concat([train_set[features], train_set[target]], axis=1),
          target=target, session_id=123, fold=10, fold_shuffle=True, use_gpu=True, verbose=False)

    hummingbird_supported = {
        'ExtraTreesRegressor',
        'RandomForestRegressor',
        'DecisionTreeRegressor',
        'GradientBoostingRegressor',
        'LGBMRegressor',
        'LinearRegression',
        'Ridge',
        'Lasso',
        'ElasticNet'
    }

    top_models = compare_models(n_select=5)
    results_df = pull()

    best_model = top_models[0]
    best_model_name = type(best_model).__name__
    best_r2 = results_df.iloc[0]['R2']
    best_mae = results_df.iloc[0]['MAE']
    best_mse = results_df.iloc[0]['MSE']

    logging.info(f"🔍 First model suggested: {best_model_name} (R2={best_r2:.4f}, MAE={best_mae:.2f}, MSE={best_mse:.2f})")

    if best_model_name not in hummingbird_supported:
        logging.info(f"❌ {best_model_name} not supported by Hummingbird. Searching for alternative...")

        for i, model in enumerate(top_models):
            model_type_name = type(model).__name__
            model_r2 = results_df.iloc[i]['R2']
            model_mae = results_df.iloc[i]['MAE']
            model_mse = results_df.iloc[i]['MSE']

            if model_type_name in hummingbird_supported:
                r2_diff = abs(best_r2 - model_r2) / abs(best_r2) if best_r2 != 0 else float('inf')
                mae_diff = abs(best_mae - model_mae) / abs(best_mae) if best_mae != 0 else float('inf')
                mse_diff = abs(best_mse - model_mse) / abs(best_mse) if best_mse != 0 else float('inf')

                if r2_diff <= 0.05 and mae_diff <= 0.05 and mse_diff <= 0.05:
                    logging.info(f"✅ Switching to {model_type_name} (R2={model_r2:.4f}, MAE={model_mae:.2f}, MSE={model_mse:.2f})")
                    best_model = model
                    best_model_name = model_type_name
                    break

    logging.info(f"Hereeeeeeeeee")
    tuned_model = tune_model(best_model)
    final_model = finalize_model(tuned_model)

    logging.info(f"✅ Final model selected: {best_model_name}")
    return final_model, best_model_name




def objective(trial, X_train, y_train, X_test, y_test, model_type):
    """Objective function for Optuna hyperparameter tuning"""
    params = {}
    
    if model_type == "ExtraTreesRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        model = ExtraTreesRegressor(**params)
    
    elif model_type == "RandomForestRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        model = RandomForestRegressor(**params)
    
    elif model_type == "GradientBoostingRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
    
    elif model_type == "AdaBoostRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'random_state': 42
        }
        model = AdaBoostRegressor(**params)
    
    elif model_type == "DecisionTreeRegressor":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': 42
        }
        model = DecisionTreeRegressor(**params)
    
    elif model_type == "KNeighborsRegressor":
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        }
        model = KNeighborsRegressor(**params)
    
    elif model_type == "Lasso":
        params = {'alpha': trial.suggest_float('alpha', 0.0001, 10.0)}
        model = Lasso(**params)
    
    elif model_type == "Ridge":
        params = {'alpha': trial.suggest_float('alpha', 0.0001, 10.0)}
        model = Ridge(**params)
    
    elif model_type == "BayesianRidge":
        model = BayesianRidge()
    
    elif model_type == "LinearRegression":
        model = LinearRegression()
    
    elif model_type == "Lars":
        model = Lars()
    
    elif model_type == "LassoLars":
        model = LassoLars()
    
    elif model_type == "ElasticNet":
        params = {'alpha': trial.suggest_float('alpha', 0.0001, 10.0)}
        model = ElasticNet(**params)
    
    elif model_type == "HuberRegressor":
        model = HuberRegressor()
    
    elif model_type == "OrthogonalMatchingPursuit":
        model = OrthogonalMatchingPursuit()
    
    elif model_type == "PassiveAggressiveRegressor":
        model = PassiveAggressiveRegressor()
    
    elif model_type == "DummyRegressor":
        model = DummyRegressor()
    
    elif model_type == 'LGBMRegressor':
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'num_leaves': trial.suggest_int("num_leaves", 20, 150),
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 1.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        model = LGBMRegressor(**params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)  # Minimizing MSE
    return mse

def train_optuna_model(train_set, test_set, features, target, model_type):
    
    """Train the same model type as PyCaret using Optuna"""
    X_train, y_train = train_set[features], train_set[target]
    X_test, y_test = test_set[features], test_set[target]

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_type), n_trials=30)
    logging.info(f"model_type : {model_type}")

    # Train final model with best params
    best_params = study.best_params
        
    model_class = globals().get(model_type)
    logging.info(f"model class : {model_class}")
    if model_class is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    final_model = model_class(**best_params)
    final_model.fit(X_train, y_train)
    
    return final_model

def extract_model(pycaret_pipeline):
    """Extracts the final estimator from a PyCaret pipeline."""
    if hasattr(pycaret_pipeline, "named_steps"):
        logging.info(f"Pipeline steps: {pycaret_pipeline.named_steps}")
        return pycaret_pipeline.named_steps.get("actual_estimator", pycaret_pipeline)
    else:
        logging.info("Not a pipeline, returning the model directly.")
        return pycaret_pipeline  # If it's already a model, return as is


def evaluate_models(model1, model2, X_test, y_test, model_name):
    """Compare two models and return the best one based on R², MAE, and MSE"""
    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)

    metrics1 = {
        "r2": r2_score(y_test, preds1),
        "mae": mean_absolute_error(y_test, preds1),
        "mse": mean_squared_error(y_test, preds1)
    }

    metrics2 = {
        "r2": r2_score(y_test, preds2),
        "mae": mean_absolute_error(y_test, preds2),
        "mse": mean_squared_error(y_test, preds2)
    }

    logging.info(f"📊 {model_name} - PyCaret Model: {metrics1}")
    logging.info(f"📊 {model_name} - Optuna Model: {metrics2}")

    # Choose best model based on R² (higher is better), and MAE/MSE (lower is better)
    if metrics1["r2"] > metrics2["r2"] and metrics1["mae"] < metrics2["mae"] and metrics1["mse"] < metrics2["mse"]:
        logging.info(f"✅ Final Model: PyCaret {model_name}")
        # # # Save the Optuna model
        # # # Convert model to GPU using Hummingbird
        # # Extract scikit-learn models from PyCaret pipelines
        # model1_sklearn = extract_model(model1)
        # hb_model = convert(model1_sklearn, 'pytorch')
        # torch.save(hb_model, os.path.join(model_dir, f"{model_name}.pt"))
        
        # # torch.save(model1, os.path.join(model_dir, f"{model_name}_optuna.pth"))
        # logging.info(f"💾 Optuna Model saved: {model_name}.pt")
        return model1
    else:
        logging.info(f"✅ Final Model: Optuna {model_name}")
        # # # Save the Optuna model
        # model2_sklearn = extract_model(model2)
        # hb_model = convert(model2_sklearn, 'pytorch')
        # torch.save(hb_model, os.path.join(model_dir, f"{model_name}.pt"))
        
        # # torch.save(model2, os.path.join(model_dir, f"{model_name}_optuna.pth"))
        # logging.info(f"💾 Optuna Model saved: {model_name}.pt")
        return model2
    
def train_and_save_model(model,model_name,model_dir):
    try:
        # Attempt conversion to PyTorch
        pytorch_model = convert(model, 'pytorch')
        save_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(pytorch_model, save_path)
        logging.info(f"💾 Model successfully converted and saved as PyTorch: {save_path}")
    except Exception as e:
        # If conversion fails, save as a scikit-learn model
        save_path = os.path.join(model_dir, f"{model_name}.pth")
        # joblib.dump(model, save_path)
        torch.save(model,save_path)
        logging.warning(f"⚠️ PyTorch conversion failed. Saved model using torch: {save_path}")
        logging.warning(f"Conversion error: {e}")
    return None
    


if __name__ == "__main__":
    # Ensure the model directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    config = load_config()
    df = load_data(config)
    df['duration_sec'] = df['duration']/1000
    
    df = df[(df['avg_speed'] >= 10) & (df['avg_speed'] <= 60)]


    if df is not None and len(df) >= 40000:
        hypothesis_results = perform_hypothesis_tests(df)
        logging.info(f"✅ Hypothesis Testing Completed: {hypothesis_results}")

        distance_features = [k for k, v in hypothesis_results['distance'].items() if v.get('significant', False)]
        duration_features = [k for k, v in hypothesis_results['duration_sec'].items() if v.get('significant', False)]
        avg_speed_features = [k for k, v in hypothesis_results['avg_speed'].items() if v.get('significant', False)]

        logging.info(f"🚀 Distance Features: {distance_features}")
        logging.info(f"⏳ Duration Features: {duration_features}")
        logging.info(f"🚀 Avg Speed Features: {avg_speed_features}")
        
        if not any([distance_features, duration_features, avg_speed_features]):
            logging.warning("⚠️ No significant features found for any model. Exiting training process.")
            exit(1)
            
        train_set, test_set = preprocess_data(df)
        logging.info(f"train set : {train_set}")
        logging.info(f"test set : {test_set}")

        # Train PyCaret distance model
        distance_model_pycaret, best_model_name = train_pycaret_model(train_set, test_set, distance_features, 'distance', "distance_model", model_dir)
        # Train Optuna model using same model type
        distance_model_optuna = train_optuna_model(train_set, test_set, distance_features, 'distance', best_model_name)
        # Compare and select the best model
        distance_final_model = evaluate_models(distance_model_pycaret, distance_model_optuna, test_set[distance_features], test_set['distance'], "distance_model")
        # Extract scikit-learn models from PyCaret pipelines
        distance_sklearn = extract_model(distance_final_model)
        # distance_model = convert(distance_sklearn, 'pytorch')
        # torch.save(distance_model, os.path.join(model_dir, "distance_model.pt"))
        train_and_save_model(distance_sklearn,'distance_model',model_dir)
        logging.info(f"💾 Model saved: distance")
        
        
        # Train PyCaret duration model
        duration_model_pycaret, duration_best_model_name = train_pycaret_model(train_set, test_set, duration_features, 'duration_sec', "duration_model", model_dir)
        # Train Optuna model using same model type
        duration_model_optuna = train_optuna_model(train_set, test_set, duration_features, 'duration_sec', duration_best_model_name)
        # Compare and select the best model
        duration_final_model = evaluate_models(duration_model_pycaret, duration_model_optuna, test_set[duration_features], test_set['duration_sec'], "duration_model")
        
        # Extract scikit-learn models from PyCaret pipelines
        duration_sklearn = extract_model(duration_final_model)
        # duration_model = convert(duration_sklearn, 'pytorch')
        # torch.save(duration_model, os.path.join(model_dir, "duration_model.pt"))
        train_and_save_model(duration_sklearn,'duration_model',model_dir)
        logging.info(f"💾 Model saved: duration")
        
        # # Train PyCaret avg speed model
        avg_speed_model_pycaret, avg_speed_best_model_name = train_pycaret_model(train_set, test_set, avg_speed_features, 'avg_speed', "avg_speed_model", model_dir)

        # Train Optuna model using same model type
        avg_speed_model_optuna = train_optuna_model(train_set, test_set, avg_speed_features, 'avg_speed', avg_speed_best_model_name)

        # Compare and select the best model
        avg_speed_final_model = evaluate_models(avg_speed_model_pycaret, avg_speed_model_optuna, test_set[avg_speed_features], test_set['avg_speed'], "avg_speed_model")
        
        print(type(avg_speed_final_model))
        # Extract scikit-learn models from PyCaret pipelines
        avg_speed_sklearn = extract_model(avg_speed_final_model)
        # avg_speed_model = convert(avg_speed_sklearn, 'pytorch')
        # torch.save(avg_speed_model, os.path.join(model_dir, "duration_model.pt"))
        train_and_save_model(avg_speed_sklearn,'avg_speed_model',model_dir)
        logging.info(f"💾 Model saved: avg speed")
        

    else:
        logging.warning("⚠️ Not enough data for training")
