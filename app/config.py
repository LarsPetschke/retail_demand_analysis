# app/config.py

PREPARED_DATA_FILE  = "data/preprocessed_data/train_guayas_model_ready.csv"
XGB_ARCHIVE_DIR     = "model/xgb/archive"
LSTM_ARCHIVE_DIR    = "model/lstm/archive"
SCALER_ARCHIVE_DIR  = "model/scaler/archive"
MLFLOW_XGB_DEFAULT  = "mlruns/xgb_default"
MLFLOW_XGB_HYPER    = "mlruns/xgb_hyperopt"
MLFLOW_LSTM_SEQ     = "mlruns/lstm_seq2seq"
SEQ_LEN             = 90


# Directory paths for data and model files
DATA_PATH = "data/"  # Path to the directory containing the raw data files
MODEL_PATH = 'model/'  # Path to the directory containing the model files
GLOBAL_XGB_MODEL = "xgb_global.pkl"

# Filenames global LSTMs
LSTM_GLOBAL_MODEL            = "lstm_global.keras"
LSTM_GLOBAL_SCALER           = "lstm_global_scaler.pkl"
LSTM_OPTIMAL_MODEL           = "lstm_optimized_model.keras"
LSTM_OPTIMAL_SCALER          = "lstm_optimized_scaler.pkl"
LSTM_GLOBAL_MODEL_SEQ2SEQ    = "lstm_global_model_seq2seq.keras"
LSTM_GLOBAL_SCALER_SEQ2SEQ   = "lstm_global_scaler_seq2seq.pkl"
# LSTM_GLOBAL_SEQ2SEQ_MODEL    = "lstm_global_seq2seq.keras"
LSTM_GLOBAL_SEQ2SEQ_MODEL    = "lstm_seq2seq_store24_item220435.keras"
# LSTM_GLOBAL_SEQ2SEQ_SCALER   = "lstm_global_seq2seq_scaler.pkl"
LSTM_GLOBAL_SEQ2SEQ_SCALER   = "scaler_seq2seq_store24_item220435.pkl"
LSTM_DEFAULT_MODEL           = "lstm_default_store24_item220435.keras"
LSTM_DEFAULT_SCALER          = "lstm_default_scaler_store24_item220435.pkl"
# STM_GLOBAL_MODEL            = "model_summary.txt"

# Google Drive file IDs for each dataset
# Replace these with actual file IDs from Google Drive where the datasets are stored
your_file_id_for_stores_csv = '1Pu77cFsG2Ov6-0xKcqLvpLENuE1sOgWS'  # ID for stores data CSV
your_file_id_for_items_csv = '1vyGLmw1z2873hBausGTSA164_yfhRW-t'  # ID for items data CSV
your_file_id_for_transactions_csv = '1-Az-VKqpRqkQ8ScD5VEo1fB4YJhsFCPe'  # ID for transactions data CSV
your_file_id_for_oil_csv = '1a4hOgjeuT23qDXDGaBhjRctOu5wYnzzy'  # ID for oil prices data CSV
your_file_id_for_holidays_csv = '1y-kIjUmrVE5hXOwTynryz9ddSk91dHAA'  # ID for holidays data CSV
your_file_id_for_train_csv = '1-7S9_L8r9_fFZo9a-5htkEg8HE0rTDiM'  # ID for training data CSV

# Google Drive links for each dataset
# These links are dynamically constructed using the file IDs, making it easy to download the data
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",  # Link for stores data
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",  # Link for items data
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",  # Link for transactions data
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",  # Link for oil prices data
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",  # Link for holidays data
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"  # Link for training data
}

# Google Drive link for the model
# Replace the file ID below with the actual file ID for the XGBoost model saved in Google Drive
your_file_id_for_xgboost_model_xgb = "1DrOVG_WNUvOcpbF_EBKUZv0pOWwklRh_"  # ID for the XGBoost model file

# Google Drive link for the model file
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"  # Link for the XGBoost model
}

from datetime import datetime

APP_TITLE = "Sales Forecast for Guayas Region"

DATA_PATH = "data/preprocessed_data/train_guayas_prepared.csv"

# Configuration values
CUTOFF_DATE = datetime(2013, 12, 31)
FORECAST_END = "2014-03-31"

# Features used in the model
FEATURE_COLS = [
    'store_nbr','item_nbr','unit_sales','onpromotion',
    'day_of_week','month','year','unit_sales_7d_avg',
    'family_code','lag_1','lag_7','rolling_mean_7'
]
FEATURES = [
    "store_nbr", "item_nbr", "onpromotion", "day_of_week", "month",
    "unit_sales_7d_avg", "lag_1", "lag_7", "rolling_mean_7"
]
TARGET = "unit_sales"



# Hyperparameter space
HYPEROPT_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate_range": (0.01, 0.1, 0.2, 0.3),
    "n_estimators": [20, 50, 100]
}

LSTM_PARAMS = {
    "lstm_units": 16,
    "dropout_rate": 0.1,
    "dense_units": 32,

    "epochs": 10,
    "batch_size": 16,
    "patience": 2
}
# LSTM_Params
LSTM_PARAMS_1 = {
    "lstm_units": [16, 32, 64],
    "dropout_rate": [0.1, 0.2, 0.3],
    "dense_units": [16, 32, 64],
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "rmsprop"],
    "loss": ["mse", "mae"],
    "metrics": ["mae", "mse"],
    "epochs": [10, 20, 30],
    "batch_size": [16, 32, 64],
    "patience": [2, 3, 4]
}

# LSTM input shape
