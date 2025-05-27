# --- Standard Libraries ---
import os
import sys
from pathlib import Path
from datetime import timedelta

# --- Data Handling & Visualization ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- ML Evaluation ---
from sklearn.metrics import mean_absolute_error, r2_score

# --- Web App Framework ---
import streamlit as st

# --- Add project root to path dynamically ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- App Imports ---
import config as cfg
from model.model_utils import run_forecast

# --- Streamlit Page Config ---
st.set_page_config(page_title=cfg.APP_TITLE, layout='centered')
st.sidebar.title(cfg.APP_TITLE)

# --- Sidebar Help Text ---
st.sidebar.markdown("""
### ❓ Help  
Select a store, item, and model.  
The forecast will update automatically.
""")

# --- File Paths ---
PREPARED_DATA_PATH = project_root / cfg.PREPARED_DATA_FILE
XGB_ARCHIVE = project_root / cfg.XGB_ARCHIVE_DIR
LSTM_ARCHIVE = project_root / cfg.LSTM_ARCHIVE_DIR
SCALER_ARCHIVE = project_root / cfg.SCALER_ARCHIVE_DIR

# --- Load Data (Cached) ---
@st.cache_data
def load_data():
    return pd.read_csv(PREPARED_DATA_PATH, parse_dates=['date'])

df = load_data()

# --- Feature Setup ---
FEATURE_COLS = cfg.FEATURE_COLS
TARGET = cfg.TARGET
CUTOFF = pd.to_datetime(cfg.CUTOFF_DATE)

# --- Sidebar Selections ---
store = st.sidebar.selectbox('Store ID', sorted(df['store_nbr'].unique()))
item = st.sidebar.selectbox('Item ID', sorted(df[df['store_nbr'] == store]['item_nbr'].unique()))

model_options = {
    'XGB Global': 'xgb_global',
    'XGB Hyperopt': 'xgb_hyperopt',
    'LSTM Global': 'lstm_global'
}
model_choice = st.sidebar.radio('Model', model_options.keys())
model_key = model_options[model_choice]

# --- Filter Data ---
sub = df[(df['store_nbr'] == store) & (df['item_nbr'] == item)].sort_values('date')


# --- Forecast Dispatcher ---
def forecast_model():
    if model_key == 'xgb_global':
        return forecast_xgb_global(sub)

    elif model_key == 'xgb_hyperopt':
        try:
            return run_forecast(df, store, item)
        except Exception as e:
            st.error(f"Forecast error: {e}")
            return None

    elif model_key == 'lstm_global':
        return forecast_lstm_global()


# --- Forecast Functions ---
def forecast_xgb_global(sub_df):
    model_path = XGB_ARCHIVE / cfg.GLOBAL_XGB_MODEL
    if not model_path.exists():
        st.warning(f"No model found at: {model_path}")
        return None

    data = joblib.load(model_path)
    model, feats = data['model'], data['features']

    test_df = sub_df[sub_df['date'] > CUTOFF]
    if test_df.empty:
        st.warning("No data available after cutoff date.")
        return None

    X_test = test_df[feats].astype('float64')
    preds = np.expm1(model.predict(X_test))

    return pd.DataFrame({
        'date': test_df['date'],
        'actual': test_df[TARGET].values,
        'forecast': preds
    })


def forecast_lstm_global():
    try:
        model_path = LSTM_ARCHIVE / cfg.LSTM_GLOBAL_MODEL
        scaler_path = SCALER_ARCHIVE / cfg.LSTM_GLOBAL_SCALER

        if not model_path.exists() or not scaler_path.exists():
            st.warning(f"Model or scaler missing:\n{model_path}\n{scaler_path}")
            return None

        # Reload utility in case of Streamlit hot reload
        import importlib
        import model.model_utils
        importlib.reload(model.model_utils)
        from model.model_utils import forecast_with_saved_lstm

        return forecast_with_saved_lstm(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            df=df,
            store_nbr=store,
            item_nbr=item
        )

    except Exception as e:
        st.error(f"LSTM Forecast error: {e}")
        return None


# --- Execute Forecast ---
result = forecast_model()
if result is None:
    st.stop()

# --- Compute Metrics ---
test_result = result.dropna()
if not test_result.empty:
    r2 = r2_score(test_result['actual'], test_result['forecast'])
    mae = mean_absolute_error(test_result['actual'], test_result['forecast'])
else:
    r2 = mae = float('nan')

# --- Page Title & Metrics ---
st.title(cfg.APP_TITLE)
st.markdown(f"**Store:** {store} | **Item:** {item} | **Model:** {model_choice}  \n"
            f"**R²:** {r2:.4f} | **MAE:** {mae:.2f}")

# --- Full Time Series Chart ---
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(sub['date'], sub[TARGET], label='Unit Sales', linewidth=2)
ax1.plot(result['date'], result['forecast'], 'r--', label='Forecast', linewidth=2)
ax1.axvline(CUTOFF, color='orange', linestyle='--', label='Cutoff Date')
ax1.set_title("Unit Sales Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Unit Sales")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Forecast vs. Actual Chart ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(result['date'], result['actual'], label='Actual', linewidth=2)
ax2.plot(result['date'], result['forecast'], 'r--', label='Forecast', linewidth=2)
ax2.set_title("Forecast vs. Actual (Test Period)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Unit Sales")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# --- Forecast Table ---
st.markdown("### 📊 Forecast Table")
st.dataframe(result[['date', 'actual', 'forecast']].round(2))
