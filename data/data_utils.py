# data/data_utils.py

import pandas as pd  # Import pandas for data manipulation and analysis
import os  # Import os to interact with the file system
import gdown  # Import gdown to download files from Google Drive
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS  # Import paths and links for data files
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder to encode categorical features

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    # Check if the file already exists locally
    if not os.path.exists(file_path):
        # If it doesn't exist, download the file from Google Drive using gdown
        gdown.download(url, file_path, quiet=False)
    else:
        # If the file already exists, print a message
        print(f"{file_path} already exists.")

def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""
    
    # Define the paths for all the required data files
    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv"  # Path for training data
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])  # Stores data
    df_items = pd.read_csv(files["items"])  # Items data
    df_transactions = pd.read_csv(files["transactions"])  # Transactions data
    df_oil = pd.read_csv(files["oil"])  # Oil prices data
    df_holidays = pd.read_csv(files["holidays_events"])  # Holidays and events data
    
    # Load data only for stores in 'Pichincha' region
    # Get the list of store IDs for the state 'Pichincha'
    store_ids = df_stores[df_stores['state'] == 'Pichincha']['store_nbr'].unique()
    # Select the same items as for "Classical methods":
    item_ids = [564533, 838216, 582865, 364606]
    # Select data before April 2014
    max_date = '2014-04-01'

    # Initialize an empty list to hold filtered chunks
    filtered_chunks = []

    # Define the chunk size (number of rows per chunk)
    chunk_size = 10 ** 6  # Adjust based on your system's memory capacity

    # Read the CSV file in chunks
    for chunk in pd.read_csv(files["train"], chunksize=chunk_size):
        # Filter the chunk for the desired store IDs
        chunk_filtered = chunk[
            (chunk['store_nbr'].isin(store_ids)) & 
            (chunk['item_nbr'].isin(item_ids)) & 
            (chunk['date'] < max_date)
        ]
        # Append the filtered chunk to the list
        filtered_chunks.append(chunk_filtered)
        # Optional: Delete the chunk to free up memory
        del chunk

    # Concatenate all filtered chunks into a single DataFrame
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)

    # Clean up to free memory
    del filtered_chunks

    # Group by date and aggregate sales
    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date']).sum()['unit_sales'].reset_index()

    # Return all the loaded DataFrames
    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_filtered

def preprocess_input_data(store_id, item_id, split_date, df_stores, df_items, df_filtered):
    """Preprocesses input data into a format suitable for model prediction."""
    

    # Convert the 'date' column to datetime format for easy manipulation
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    split_date = pd.to_datetime(split_date)  # Convert the split_date to datetime

    # Get the minimum and maximum dates in the dataset to create a full date range
    min_date = df_filtered['date'].min()
    max_date = df_filtered['date'].max()
    print("Before filtering", min_date.date(), max_date.date())

    # Filter the dataset to only include dates after the specified split date
    df_filtered = df_filtered[df_filtered['date'] >= split_date]  # Filter rows by date
    
    # Group by store, item, and date, then aggregate (sum) the unit_sales for each group
    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date']).sum()['unit_sales'].reset_index()

    # Create a full date range covering all days between the min and max dates
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

		# Create a DataFrame with all (store, item, date) combinations
    store_item_combinations = df_filtered[['store_nbr', 'item_nbr']].drop_duplicates()
    all_combinations = store_item_combinations.merge(full_date_range, how='cross')
    
    # Merge with original data to fill missing dates
    df_filled = all_combinations.merge(df_filtered, on=['store_nbr', 'item_nbr', 'date'], how='left')
    
    # Fill missing sales values with 0
    df_filled['unit_sales'] = df_filled['unit_sales'].fillna(0)

    # Feature engineering: extract date-related features
    df_filled['month'] = df_filled['date'].dt.month  # Extract the month from the date
    df_filled['day'] = df_filled['date'].dt.day  # Extract the day from the date
    df_filled['weekofyear'] = df_filled['date'].dt.isocalendar().week  # Extract the ISO week of the year
    df_filled['dayofweek'] = df_filled['date'].dt.dayofweek  # Extract the day of the week (0=Monday, 6=Sunday)
    
    # Create rolling features for unit_sales (7-day rolling mean and standard deviation)
    df_filled['rolling_mean'] = df_filled['unit_sales'].rolling(window=7).mean()
    df_filled['rolling_std'] = df_filled['unit_sales'].rolling(window=7).std()

    # Create lag features (sales from the previous day, previous week)
    df_filled['lag_1'] = df_filled['unit_sales'].shift(1)  # Sales from the previous day
    df_filled['lag_7'] = df_filled['unit_sales'].shift(7)  # Sales from 7 days ago
    df_filled['lag_30'] = df_filled['unit_sales'].shift(30)  # Sales from 30 days ago

    # Drop any rows with NaN values after creating lag features (for rows without enough data)
    df_filled.dropna(inplace=True)

    # Merge the filled DataFrame with store and item data to include more information
    df_filled = df_filled.merge(df_stores, on='store_nbr', how='left').merge(df_items, on='item_nbr', how='left')

    # Encode categorical columns with LabelEncoder to convert them into numeric format
    for col in ['city', 'state', 'type', 'family', 'class']:  # List of categorical columns to encode
        le = LabelEncoder()  # Initialize the label encoder
        df_filled[col] = le.fit_transform(df_filled[col])  # Apply the encoder to the column

    # Sort the final DataFrame by store number, item number, and date
    df_filled = df_filled.sort_values(by=['store_nbr', 'item_nbr', 'date'])

    # Return the preprocessed and feature-engineered DataFrame
    return df_filled