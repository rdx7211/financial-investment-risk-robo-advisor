import joblib
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.feature_preprocessing.feature_transformation import log_transformation,  encode_categorical_features
from src.portfolio_optimization.markowitz import calculate_portfolio_statistics as calc_portfolio_stats, optimize_portfolio as opt_portfolio


data = pd.read_csv(os.path.join(project_root, "data", "raw", "stock_data.csv"), index_col='Date')

def load_model(model_path):
    """
    Loads the trained model from the specified path.
    """
    return joblib.load(model_path)

def preprocess_input_data(df, columns_to_log_transform):
    """
    Applies log transformation and standard scaling to input data.
    """
    df = log_transformation(df, columns_to_log_transform)
    df = encode_categorical_features(df)
    
    return df

def calculate_portfolio_statistics(sample_data):
    """
    Calculates daily returns, mean historical returns, and covariance matrix.
    """
    # Calculate daily returns
    daily_returns = sample_data.pct_change().dropna()
    
    # Calculating the annualized expected returns and the annualized sample covariance matrix
    mean_returns = expected_returns.mean_historical_return(sample_data)
    covariance = risk_models.sample_cov(sample_data)
    
    return daily_returns, mean_returns, covariance

def optimize_portfolio(sample_data, risk_tolerance):
    """
    Optimizes portfolio allocation based on risk tolerance and returns cumulative portfolio performance.
    """
    # Normalize risk tolerance to range 0-1
    risk_tolerance = risk_tolerance / 100.0
    
    # Compute portfolio statistics
    _, mean_returns, covariance = calculate_portfolio_statistics(sample_data)
    
    # Initialize Efficient Frontier
    ef = EfficientFrontier(mean_returns, covariance)
    
    # Apply threshold condition
    if risk_tolerance > 0.5:
        ef.max_sharpe()  # Aggressive strategy
    else:
        ef.min_volatility()  # Conservative strategy
    
    clean_weights = ef.clean_weights()
    
    # Compute cumulative returns using optimized weights
    weighted_returns = (sample_data.pct_change().dropna() @ np.array(list(clean_weights.values())))
    cumulative_returns = (1 + weighted_returns).cumprod()
    
    return clean_weights, cumulative_returns

def make_predictions(input_data, model_path, columns_to_log_transform):
    """
    Takes user input, preprocesses it, and returns predictions.
    """
    model = load_model(model_path)
    
    # Convert input into a DataFrame
    df = pd.DataFrame([input_data])
    df = preprocess_input_data(df, columns_to_log_transform)
    
    predictions = model.predict(df)
    return predictions[0]

if __name__ == "__main__":
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    columns_to_log_transform = ['INCOME', 'NETWORTH']
    le_columns = ['EDUCATION_LEVEL', 'MARITAL_STATUS', 'OCCUPATION_CATEGORY', 'SPENDING_VS_INCOME']
    
    # User input for prediction
    user_input = {
         'AGE': 50,
         'EDUCATION_LEVEL': 'high_school',
         'MARITAL_STATUS': 'married',
         'NO_OF_KIDS': 4,
         'OCCUPATION_CATEGORY': 'Senior_level',
         'INCOME': 70000,
         'RISK_LEVEL': 3,
         'SPENDING_VS_INCOME': 'spends_more',
         'SPENDING_LEVEL': 5,
         'NETWORTH': 300000
    }
    
    predicted_risk_tolerance = make_predictions(user_input, model_path, columns_to_log_transform)
    print("Predicted Risk Tolerance:", predicted_risk_tolerance)
    
    # Define sample stocks
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    sample_data = data[sample_stocks].copy()
    
    # Calculate portfolio metrics
    daily_returns, mean_returns, covariance = calculate_portfolio_statistics(sample_data)
    
    # Optimize portfolio
    optimized_weights, cumulative_returns = optimize_portfolio(sample_data, predicted_risk_tolerance)
    
    print("Optimized Portfolio Weights:", optimized_weights)
    print("Cumulative Returns:\n", cumulative_returns.head())