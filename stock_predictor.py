import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Deep Learning Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def fetch_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_tabular_data(data):
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df['Target'] = df['Close'].shift(-1)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'Daily_Return']
    X = df[features]
    y = df['Target']
    return X, y, df

def prepare_lstm_data(df, look_back=60):
    print("Preparing sequential data for LSTM...")
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    # We will use only the 'Close' price for the basic LSTM
    close_prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, data.index[look_back:]

def train_and_evaluate_baselines(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    print("\n[Linear Regression] MSE: {:.4f}, MAE: {:.4f}".format(
        mean_squared_error(y_test, lr_pred), mean_absolute_error(y_test, lr_pred)))
    print("[Random Forest] MSE: {:.4f}, MAE: {:.4f}".format(
        mean_squared_error(y_test, rf_pred), mean_absolute_error(y_test, rf_pred)))
        
    return lr_pred, rf_pred, y_test

def build_and_train_lstm(X, y):
    # Split chronologically
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining LSTM model... (Train size: {len(X_train)}, Test size: {len(X_test)})")
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model (epochs=10 for faster execution during prototyping)
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=1)
    
    lstm_predictions_scaled = model.predict(X_test)
    
    return model, lstm_predictions_scaled, y_test, split_idx

def plot_all_predictions(ticker, df_baselines, y_test_base, lr_pred, rf_pred, 
                         df_lstm_dates, y_test_lstm, lstm_pred):
    plt.figure(figsize=(16, 8))
    
    # Plot Actual Prices from baseline test set for reference
    plt.plot(y_test_base.index, y_test_base.values, label='Actual Price', color='black', linewidth=2)
    
    # Plot Baseline Predictions
    plt.plot(y_test_base.index, lr_pred, label='Linear Regression', color='red', alpha=0.6)
    plt.plot(y_test_base.index, rf_pred, label='Random Forest', color='green', alpha=0.6)
    
    # Plot LSTM Predictions (Note: The dates might slightly differ because of the look_back window)
    # We use the dates corresponding to the LSTM test set
    lstm_test_dates = df_lstm_dates[-len(lstm_pred):]
    plt.plot(lstm_test_dates, lstm_pred, label='Deep Learning (LSTM)', color='blue', linewidth=2)
    
    plt.title(f'{ticker} Stock Price Prediction: Baselines vs Deep Learning')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_plot_advanced.png')
    print("\nAdvanced plot saved to prediction_plot_advanced.png")
    # plt.show()

def main():
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2023-01-01'
    
    data = fetch_data(ticker, start_date, end_date)
    
    print("\n--- 1. Evaluating Traditional Baselines ---")
    X_base, y_base, df_base = preprocess_tabular_data(data)
    lr_pred, rf_pred, y_test_base = train_and_evaluate_baselines(X_base, y_base)
    
    print("\n--- 2. Evaluating Deep Learning (LSTM) ---")
    look_back = 60 # Number of previous days used to predict the next day
    X_lstm, y_lstm, scaler, lstm_dates = prepare_lstm_data(data, look_back)
    lstm_model, lstm_pred_scaled, y_test_lstm_scaled, split_idx = build_and_train_lstm(X_lstm, y_lstm)
    
    # Inverse transform LSTM predictions and actuals back to original price scale
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    y_test_lstm = scaler.inverse_transform(y_test_lstm_scaled.reshape(-1, 1))
    
    print("\n[LSTM] MSE: {:.4f}, MAE: {:.4f}".format(
        mean_squared_error(y_test_lstm, lstm_pred), mean_absolute_error(y_test_lstm, lstm_pred)))
    
    # 3. Plot Results
    print("\n--- 3. Generating Visualization ---")
    plot_all_predictions(ticker, df_base, y_test_base, lr_pred, rf_pred, 
                         lstm_dates, y_test_lstm, lstm_pred)

if __name__ == '__main__':
    main()
