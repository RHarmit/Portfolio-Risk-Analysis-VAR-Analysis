import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Define Client Portfolio
portfolio = {
    'AAPL': 50,  # 50 shares of Apple
    'MSFT': 30,  # 30 shares of Microsoft
    'GOOGL': 20,  # 20 shares of Google
}

# Step 2: Fetch Historical Stock Prices (Last 1 Year)
tickers = list(portfolio.keys())
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")['Adj Close']

# Step 3: Calculate Daily Returns
returns = data.pct_change().dropna()

# Step 4: Portfolio Value Calculation
weights = np.array([portfolio[ticker] for ticker in tickers])
current_prices = data.iloc[-1]
portfolio_value = sum(portfolio[ticker] * current_prices[ticker] for ticker in tickers)
weights = weights / portfolio_value  # Normalize Weights

# Step 5: Compute Portfolio VaR (Monte Carlo & Historical)
simulations = 10000
days = 1  # One-day VaR

# Monte Carlo Simulation
simulated_returns = np.random.normal(returns.mean(), returns.std(), (simulations, len(tickers)))
portfolio_returns_simulated = np.dot(simulated_returns, weights)
VaR_95 = np.percentile(portfolio_returns_simulated, 5) * portfolio_value

# Historical VaR
VaR_hist_95 = np.percentile(returns.dot(weights) * portfolio_value, 5)

# Step 6: Generate Risk Report
print(f"🔹 Portfolio Value: ${portfolio_value:.2f}")
print(f"🔸 Monte Carlo VaR (95%): ${abs(VaR_95):.2f}")
print(f"🔸 Historical VaR (95%): ${abs(VaR_hist_95):.2f}")

# Step 7: Visualizing Portfolio Risk
plt.figure(figsize=(10, 5))
sns.histplot(portfolio_returns_simulated * portfolio_value, bins=50, kde=True, color='blue')
plt.axvline(x=VaR_95, color='red', linestyle='dashed', label=f'VaR 95%: ${abs(VaR_95):.2f}')
plt.xlabel("Portfolio Loss ($)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulated Portfolio VaR")
plt.legend()
plt.show()

# Step 8: Alert for Risk Threshold Breach
risk_threshold = portfolio_value * 0.02  # Example: Alert if risk exceeds 2% of portfolio
if abs(VaR_95) > risk_threshold:
    print("⚠️ Risk Alert: Portfolio risk exceeds threshold! Review exposure.")
