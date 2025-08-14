import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']

end_date = datetime.today()

start_date = end_date - timedelta(days=365 * 5)
# print(f"Fetching data from {start_date} to {end_date}")
adj_close_df = pd.DataFrame()

for ticker in tickers: 
    data = yf.download(ticker, start=start_date, end=end_date) 
    adj_close_df[ticker] = data['Close']

log_returns_df = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

cov_matrix = log_returns_df.cov() * 252  # Annualize the covariance matrix

def standard_deviation(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def expected_return(weights, log_returns_df):
    return np.sum(log_returns_df.mean() * weights) * 252  # Annualize the expected return

def sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate=0.01):
    exp_return = expected_return(weights, log_returns_df)
    std_dev = standard_deviation(weights, cov_matrix)
    return (exp_return - risk_free_rate) / std_dev

def neg_sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate=0.01):
    return -sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate)

risk_free_rate = 0.02

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1 }  # Weights must sum to 1]
bounds = [(0, 0.5) for _ in range(len(tickers))]  # No short selling, max 50% in each asset

initial_weights = np.array([1/len(tickers)] * len(tickers))  # Equal initial weights
print (f"Initial Weights: {initial_weights}")

optimized_result = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns_df, cov_matrix, risk_free_rate), method ='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimized_result.x
print(f"Optimal Weights: {optimal_weights}")