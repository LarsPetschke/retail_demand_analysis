# model/model_utils.py

import pickle  # Import pickle to handle model loading (not used in this version, but included for future use)
import xgboost as xgb  # Import XGBoost library for loading and using the XGBoost model
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS  # Import paths and links for the model files
from data.data_utils import download_file  # Import a utility function for downloading files from Google Drive

def load_model(model_path=MODEL_PATH):
    """
    Downloads necessary data from Google Drive and loads a pre-trained model.
    
    This function checks if the model file exists locally, and if not, it downloads the model from 
    the specified Google Drive link. It then loads the model into memory using XGBoost's API.
    """
    # Define paths to model files - specifying the model file to be used
    files = {
        "xgboost_model": f"{model_path}model.xgb"  # Path to the XGBoost model file
    }

    # Download model files from Google Drive if they do not exist locally
    for key, file_path in files.items():
        # Calls the download_file function to fetch the model from Google Drive
        download_file(file_path, GOOGLE_DRIVE_LINKS_MODELS[key])
   
    # Load the pre-trained XGBoost model from the downloaded .xgb file
    # Initialize the XGBoost model (XGBRegressor is commonly used for regression tasks)
    xgboost_model = xgb.XGBRegressor() 
    # Load the saved model from the specified file path
    xgboost_model.load_model(files["xgboost_model"])
    
    # Return the loaded model so it can be used for predictions
    return xgboost_model

def predict(model, input_data):
    """
    Runs prediction on input data using the pre-trained model.
    
    This function takes the pre-trained model and input data, processes the data (removes 
    unnecessary columns), and then performs the prediction using the model. It returns the predictions.
    """
    # Drop the original 'date' column as it is not needed for making predictions
    input_data = input_data.drop(columns=['date'])  # Remove the 'date' column

    # Drop the 'unit_sales' column, as it's the target variable and not needed as input for prediction
    input_data = input_data.drop(columns=['unit_sales'])  # Remove the 'unit_sales' column

    # Use the model to predict the sales based on the remaining input data
    prediction = model.predict(input_data)
    
    # Return the prediction results
    return prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def run_forecast(
    df: pd.DataFrame,
    store_id: int,
    item_id: int,
    cutoff: str = "2013-12-31",
    max_evals: int = 15
) -> pd.DataFrame:
    """
    Filter df by store & item, feature-engineer, tune with Hyperopt + XGB,
    train final model, and return a forecast DataFrame with columns:
    [date, actual, forecast]
    """
    # Filter data
    subset = df[(df["store_nbr"] == store_id) & (df["item_nbr"] == item_id)].copy()
    if len(subset) < 100:
        raise ValueError("Not enough data for this store/item combination.")

    subset = subset.sort_values("date")
    cutoff_date = pd.to_datetime(cutoff)
    train = subset[subset["date"] <= cutoff_date]
    test = subset[subset["date"] > cutoff_date]

    # Feature columns
    features = [
        'onpromotion', 'day_of_week', 'month', 'year', 'unit_sales_7d_avg',
        'lag_1', 'lag_7', 'rolling_mean_7'
    ]
    target = 'unit_sales'

    X_train, y_train = train[features], np.log1p(train[target])
    X_test, y_test = test[features], test[target]

    # Define search space
    space = {
        'max_depth': hp.choice('max_depth', range(3, 8)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'n_estimators': hp.choice('n_estimators', range(50, 300, 50)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
    }

    def objective(params):
        model = xgb.XGBRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        score = mean_absolute_error(y_test, pred)
        return {'loss': score, 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_model = trials.best_trial['result']['model']

    # Forecast
    pred_log = best_model.predict(X_test)
    forecast = np.expm1(pred_log)

    forecast_df = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test.values,
        "forecast": forecast
    })

    return forecast_df


import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from app import config as cfg


'''def forecast_with_saved_lstm(model_path, scaler_path, df, store_nbr, item_nbr):
    """
    Führt eine 90-Tage-Vorhersage mit einem globalen Seq2Seq-LSTM-Modell durch,
    das mehrere Features nutzt (z. B. store_nbr, item_nbr, lag-Werte usw.).
    """
    DTYPE = np.float16  # Datentyp für die Vorhersage
    FEATURES = ['unit_sales', 'onpromotion', 'lag_1'
            ]#, 'rolling_mean_7', 'store_nbr', 'item_nbr'
                
    # --- Subset für Store & Item ---
    sub = df[(df["store_nbr"] == store_nbr) & (df["item_nbr"] == item_nbr)].copy()
    sub = sub.sort_values("date")

    if len(sub) < cfg.SEQ_LEN:
        raise ValueError(f"Nicht genug Daten für Store {store_nbr}, Item {item_nbr}.")

    # --- Nur bis zum Cutoff-Datum trainieren ---
    sub_train = sub[sub["date"] <= cfg.CUTOFF_DATE]
    if len(sub_train) < cfg.SEQ_LEN:
        raise ValueError("Nicht genug Datenpunkte vor dem Cutoff.")

    # --- Modell & Scaler laden ---
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # --- Features extrahieren und skalieren ---
    last_seq = sub_train[FEATURES].values[-cfg.SEQ_LEN:]
    last_scaled = scaler.transform(last_seq).reshape(1, cfg.SEQ_LEN, len(FEATURES))

    # --- Decoder-Input für Teacher Forcing (leer initialisieren) ---
    decoder_input = np.zeros((1, cfg.SEQ_LEN, len(FEATURES)), dtype=DTYPE)

    # --- Vorhersage (nur target-Spalte) ---
    pred_scaled = model.predict([last_scaled, decoder_input])[0]  # (SEQ_LEN, 1)

    # --- Dummy für inverse_transform: nur TARGET füllen ---
    dummy = np.zeros((cfg.SEQ_LEN, len(FEATURES)))
    target_index = FEATURES.index(cfg.TARGET)
    dummy[:, target_index] = pred_scaled[:, 0]

    # Dummy-Array mit allen Features auf 0, nur unit_sales = pred_scaled
    dummy = np.zeros((cfg.SEQ_LEN, len(FEATURES)), dtype=DTYPE)
    dummy[:, 0] = pred_scaled[:, 0]  # Annahme: unit_sales ist an Index 0

    forecast = scaler.inverse_transform(dummy)[:, 0]


        # --- Forecast-Daten vorbereiten ---
    forecast_dates = pd.date_range(start=cfg.CUTOFF_DATE + pd.Timedelta(days=1), periods=cfg.SEQ_LEN)
    actuals = (
        sub[sub["date"].isin(forecast_dates)]
        .set_index("date")
        .reindex(forecast_dates)[cfg.TARGET]
        .values
    )

    return pd.DataFrame({
        "date": forecast_dates,
        "actual": actuals,
        "forecast": forecast
    })'''

def forecast_with_saved_lstm(model_path, scaler_path, df, store_nbr, item_nbr):
    """
    Führt eine 90-Tage-Vorhersage mit einem globalen Seq2Seq-LSTM-Modell durch,
    mit log1p-transformiertem Zielwert.
    """
    SEQ_LEN = cfg.SEQ_LEN
    DTYPE = np.float32
    FEATURES = ['onpromotion', 'lag_1', 'rolling_mean_7']
    
    # Subset vorbereiten
    sub = df[(df["store_nbr"] == store_nbr) & (df["item_nbr"] == item_nbr)].copy()
    sub = sub.sort_values("date")

    if len(sub) < cfg.SEQ_LEN:
        raise ValueError(f"Nicht genug Daten für Store {store_nbr}, Item {item_nbr}.")

    sub_train = sub[sub["date"] <= cfg.CUTOFF_DATE]
    if len(sub_train) < cfg.SEQ_LEN:
        raise ValueError("Nicht genug Datenpunkte vor dem Cutoff.")

    # log1p anwenden
    sub_train["target"] = np.log1p(sub_train["unit_sales"])

    # Scaler & Modell laden
    model = load_model(model_path)
    target_scaler = joblib.load(scaler_path)

    # Feature-Scaler neu fitten (alternativ kannst du auch abspeichern)
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(sub_train[FEATURES].values.astype(DTYPE))

    # Letzte Sequenz (Features)
    X_pred_enc = X_scaled[-cfg.SEQ_LEN:].reshape(1, cfg.SEQ_LEN, len(FEATURES))
    X_pred_dec = np.zeros((1, cfg.SEQ_LEN, 1), dtype=DTYPE)
    X_pred_dec[:, 0, 0] = 0  # Seed-Wert (hier: neutraler Wert)

    # Vorhersage
    pred_scaled = model.predict([X_pred_enc, X_pred_dec])[0]  # (90, 1)
    pred_log = target_scaler.inverse_transform(pred_scaled)
    forecast = np.expm1(pred_log).flatten()

    # Forecast-Zeitraum
    forecast_dates = pd.date_range(start=cfg.CUTOFF_DATE + pd.Timedelta(days=1), periods=cfg.SEQ_LEN)

    # Vergleichswerte holen (falls vorhanden)
    if "unit_sales" in sub.columns:
        actuals = (
            sub[sub["date"].isin(forecast_dates)]
            .set_index("date")
            .reindex(forecast_dates)["unit_sales"]
            .values
        )
    else:
        actuals = [np.nan] * cfg.SEQ_LEN

    return pd.DataFrame({
        "date": forecast_dates,
        "actual": actuals,
        "forecast": forecast
    })
