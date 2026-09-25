# Stock Price Predictor

This project is a stock price prediction script that evaluates and compares traditional machine learning models against a deep learning model for forecasting stock prices.

## Overview

The script `stock_predictor.py` fetches historical stock data from Yahoo Finance for a specified ticker (e.g., AAPL) and date range. It then processes the data and trains three different types of models to predict future prices:
1. **Linear Regression**: A simple baseline model.
2. **Random Forest**: A robust ensemble machine learning baseline.
3. **LSTM (Long Short-Term Memory)**: A deep learning neural network designed for sequence prediction, built with TensorFlow/Keras.

The script evaluates the models using Mean Squared Error (MSE) and Mean Absolute Error (MAE), and generates a plot comparing their predictions against the actual stock prices.

## Requirements

The required dependencies are listed in `requirements.txt`:

- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to fetch data, train the models, and generate the prediction plot:

```bash
python stock_predictor.py
```

By default, the script predicts the stock price of Apple (AAPL) from 2015-01-01 to 2023-01-01. You can modify these parameters in the `main()` function of `stock_predictor.py`.

## Output

- **Console Output**: The script will print the MSE and MAE metrics for each model during execution.
- **Plot**: The script will save a graph comparing the actual prices with the predictions from all three models as `prediction_plot_advanced.png`.
