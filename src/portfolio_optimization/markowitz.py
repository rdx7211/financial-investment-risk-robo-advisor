import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns


data = pd.read_csv(r"D:\\Investor_Risk_Tolerance_and_Robo-Advisor\\data\\raw\\stock_data.csv", index_col='Date')
data



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
        weights = ef.max_sharpe()  # Aggressive strategy
    else:
        weights = ef.min_volatility()  # Conservative strategy
    
    clean_weights = ef.clean_weights()
    
    # Compute cumulative returns using optimized weights
    weighted_returns = (sample_data.pct_change().dropna() @ np.array(list(clean_weights.values())))
    cumulative_returns = (1 + weighted_returns).cumprod()
    
    return clean_weights, cumulative_returns

# Example Usage
if __name__ == "__main__":
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    risk_tolerance = 24.67
    

    sample_data= data[sample_stocks].copy()
    
    

    # Calculate portfolio metrics
    daily_returns, mean_returns, covariance = calculate_portfolio_statistics(sample_data)

    # Optimize portfolio
    weights, cumulative_returns = optimize_portfolio(sample_data, risk_tolerance)

    print("Optimized Portfolio Weights:", weights)
    print("Cumulative Returns:\n", cumulative_returns.head())
