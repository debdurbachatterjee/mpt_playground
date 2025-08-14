import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go

# --- Portfolio Functions ---
def standard_deviation(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def expected_return(weights, log_returns_df):
    return np.sum(log_returns_df.mean() * weights) * 252

def sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate):
    exp_return = expected_return(weights, log_returns_df)
    std_dev = standard_deviation(weights, cov_matrix)
    return (exp_return - risk_free_rate) / std_dev

def neg_sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate)

# --- Streamlit UI ---
st.set_page_config(page_title="Portfolio Playground", layout="wide")
st.title("📈 Modern Portfolio Theory Playground")

# Sidebar inputs
default_tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']
tickers = st.sidebar.text_input("Tickers (comma separated)", ",".join(default_tickers)).split(",")
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365*5))
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005)
max_weight = st.sidebar.slider("Max weight per asset", 0.0, 1.0, 0.5)
rebalance_months = st.sidebar.slider("Rebalancing Frequency (months)", 1, 12, 3)
optimize_button = st.sidebar.button("Run Optimization")

if optimize_button:
    # Fetch data
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker.strip(), start=start_date, end=end_date)
        adj_close_df[ticker.strip()] = data['Close']

    log_returns_df = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns_df.cov() * 252

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, max_weight) for _ in range(len(tickers))]
    initial_weights = np.array([1/len(tickers)] * len(tickers))

    # Optimization
    optimized_result = minimize(
        neg_sharpe_ratio, 
        initial_weights,
        args=(log_returns_df, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights = optimized_result.x

    # --- Rebalancing Simulation ---
    portfolio_rebalanced = pd.Series(index=log_returns_df.index, dtype=float)
    current_weights = initial_weights.copy()
    last_rebalance_date = log_returns_df.index[0]
    rebalance_dates = [last_rebalance_date]  # store the first rebalance at start

    for current_date in log_returns_df.index:
        if (current_date - last_rebalance_date).days >= rebalance_months * 30:
            data_until_now = log_returns_df.loc[:current_date]
            cov_matrix_now = data_until_now.cov() * 252
            res = minimize(
                neg_sharpe_ratio, 
                initial_weights,
                args=(data_until_now, cov_matrix_now, risk_free_rate),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            if res.success:
                current_weights = res.x
            last_rebalance_date = current_date
            rebalance_dates.append(current_date)  # log the rebalance date

        portfolio_rebalanced[current_date] = np.dot(log_returns_df.loc[current_date], current_weights)

    portfolio_rebalanced_cum = portfolio_rebalanced.cumsum()

    # --- Portfolio performance simulation ---
    equal_weights = np.array([1/len(tickers)] * len(tickers))
    portfolio_equal = (log_returns_df @ equal_weights).cumsum()
    portfolio_optimal = (log_returns_df @ optimal_weights).cumsum()

    # --- Display weights ---
    st.subheader("Optimal Portfolio Weights")
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        # Prepare and sort data
        weights_df = pd.DataFrame({"Ticker": tickers, "Weight": optimal_weights})
        weights_df = weights_df.sort_values("Weight", ascending=True)  # sort for nicer plot

        # Plot horizontal bar chart
        fig_weights = go.Figure(go.Bar(
            x=weights_df["Weight"] * 100,
            y=weights_df["Ticker"],
            orientation='h',
            text=[f"{w:.2%}" for w in weights_df["Weight"]],
            textposition='auto',
            marker_color='skyblue'
        ))
        fig_weights.update_layout(
            title="Optimal Weights",
            xaxis_title="Weight (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with col2:
        st.metric("Optimized Sharpe", f"{sharpe_ratio(optimal_weights, log_returns_df, cov_matrix, risk_free_rate):.2f}")
        st.metric("Expected Return", f"{expected_return(optimal_weights, log_returns_df):.2%}")
        st.metric("Volatility", f"{standard_deviation(optimal_weights, cov_matrix):.2%}")

    # --- Plot portfolio growth ---
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=portfolio_equal.index, y=np.exp(portfolio_equal), name="Equal Weight Portfolio"))
    fig_growth.add_trace(go.Scatter(x=portfolio_optimal.index, y=np.exp(portfolio_optimal), name="Optimized Portfolio"))
    fig_growth.add_trace(go.Scatter(x=portfolio_rebalanced_cum.index, y=np.exp(portfolio_rebalanced_cum), name=f"Rebalanced every {rebalance_months} months"))
    # Add markers at rebalance points
    fig_growth.add_trace(go.Scatter(
        x=rebalance_dates,
        y=np.exp(portfolio_rebalanced_cum.loc[rebalance_dates]),
        mode='markers',
        marker=dict(color='red', size=8, symbol='diamond'),
        name='Rebalance Points'
    ))

    # Optional: faint vertical lines
    for date in rebalance_dates:
        fig_growth.add_vline(
            x=date,
            line_width=1,
            line_dash="dot",
            line_color="red",
            opacity=0.3
        )
    fig_growth.update_layout(title="Portfolio Growth Over Time", yaxis_title="Portfolio Value (normalized)")
    st.plotly_chart(fig_growth, use_container_width=True)
