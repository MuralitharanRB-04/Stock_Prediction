import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import pickle

# Load and preprocess data
try:
    data = pd.read_csv('data/stock.csv')
    print("Initial data shape:", data.shape)
    print("Initial columns:", data.columns.tolist())
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    if data['date'].isnull().any():
        raise ValueError("Some dates could not be parsed. Check the 'date' column format.")
    print("Date column dtype after conversion:", data['date'].dtype)
    
    # Filter data for a single stock (close prices between $13 and $25)
    data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
    print("Shape after price filtering:", data.shape)
    if data.empty:
        raise ValueError("No data remains after filtering close prices between $13 and $25.")
    
    # Filter data for 2022
    start_date = datetime(2022, 1, 3)
    end_date = datetime(2022, 12, 30)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    print("Shape after date filtering:", data.shape)
    if data.empty:
        raise ValueError("No data remains after filtering for 2022 dates.")
    
    # Create features
    data['days'] = (data['date'] - start_date).dt.days
    print("Columns after adding 'days':", data.columns.tolist())
    print("First few rows with 'days':\n", data[['date', 'days', 'close']].head())
    
    data['lag1'] = data['close'].shift(1)
    data['lag2'] = data['close'].shift(2)
    data['ma7'] = data['close'].rolling(window=7).mean()
    
    # Verify columns before dropping NaNs
    print("Columns before dropping NaNs:", data.columns.tolist())
    print("NaN counts:\n", data.isnull().sum())
    
    # Drop rows with NaN values
    data = data.dropna()
    print("Shape after dropping NaNs:", data.shape)
    if data.empty:
        raise ValueError("No data remains after dropping NaN values.")
    
    # Verify columns exist
    required_columns = ['days', 'open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")
    
    # Prepare data for model
    X = data[required_columns].values
    y = data['close'].values
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced for smaller model.pkl
    model.fit(X, y)
    
    # Calculate metrics
    r_squared = model.score(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Model R² Score: {r_squared:.3f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Save the model, start_date, and metrics
    with open('models/model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'start_date': start_date,
            'r_squared': r_squared,
            'mae': mae,
            'rmse': rmse,
            'feature_columns': required_columns
        }, f)
    
    print("Model saved as models/model.pkl")
    
except Exception as e:
    print(f"Error during processing: {str(e)}")