# Stock Market Trend Dashboard
# Run: streamlit run dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# Page config
st.set_page_config(page_title="Stock Market Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("📈 Stock Market Trend Dashboard")
st.markdown("**Financial Analysis Tool for Better Investment Decisions**")

# Sidebar for inputs
st.sidebar.header("Dashboard Controls")

# Stock selection
default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
stocks = st.sidebar.multiselect(
    "Select Stocks:",
    ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC'],
    default=default_stocks
)

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(start_date, end_date),
    max_value=end_date
)

# Data fetching function
@st.cache_data
def fetch_stock_data(symbols, start, end):
    try:
        data = yf.download(symbols, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch data
if stocks and len(date_range) == 2:
    with st.spinner("Fetching stock data..."):
        stock_data = fetch_stock_data(stocks, date_range[0], date_range[1])
    
    if stock_data is not None and not stock_data.empty:
        # Handle single vs multiple stocks
        if len(stocks) == 1:
            prices = stock_data['Close'].to_frame()
            volumes = stock_data['Volume'].to_frame()
            prices.columns = [stocks[0]]
            volumes.columns = [stocks[0]]
        else:
            prices = stock_data['Close']
            volumes = stock_data['Volume']
        
        # Dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Daily Price Trends")
            
            # Line plot for daily prices
            fig_line = go.Figure()
            for stock in stocks:
                fig_line.add_trace(go.Scatter(
                    x=prices.index,
                    y=prices[stock],
                    mode='lines',
                    name=stock,
                    line=dict(width=2)
                ))
            
            fig_line.update_layout(
                title="Stock Price Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            # Calculate metrics
            latest_prices = prices.iloc[-1]
            price_changes = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100)
            
            for stock in stocks:
                delta = f"{price_changes[stock]:.2f}%"
                st.metric(
                    label=stock,
                    value=f"${latest_prices[stock]:.2f}",
                    delta=delta
                )
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔥 Correlation Heatmap")
            
            if len(stocks) > 1:
                # Calculate correlation matrix
                corr_matrix = prices.corr()
                
                # Create heatmap
                fig_heatmap, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                plt.title('Stock Price Correlation Matrix')
                st.pyplot(fig_heatmap)
            else:
                st.info("Select multiple stocks to view correlation heatmap")
        
        with col4:
            st.subheader("📊 Monthly Trading Volume")
            
            # Prepare monthly volume data
            monthly_volumes = volumes.resample('M').mean()
            
            # Create bar plot
            fig_bar = go.Figure()
            
            for stock in stocks:
                fig_bar.add_trace(go.Bar(
                    x=monthly_volumes.index.strftime('%Y-%m'),
                    y=monthly_volumes[stock],
                    name=stock,
                    opacity=0.8
                ))
            
            fig_bar.update_layout(
                title="Average Monthly Trading Volume",
                xaxis_title="Month",
                yaxis_title="Volume",
                barmode='group',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Additional analytics section
        st.subheader("📋 Detailed Analytics")
        
        # Performance summary table
        summary_data = {
            'Stock': stocks,
            'Current Price': [f"${latest_prices[stock]:.2f}" for stock in stocks],
            'Total Return': [f"{price_changes[stock]:.2f}%" for stock in stocks],
            'Volatility': [f"{prices[stock].pct_change().std() * np.sqrt(252) * 100:.2f}%" for stock in stocks],
            'Avg Volume': [f"{volumes[stock].mean():,.0f}" for stock in stocks]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Raw data expander
        with st.expander("📊 View Raw Data"):
            st.subheader("Stock Prices")
            st.dataframe(prices.tail(10))
            
            st.subheader("Trading Volumes")
            st.dataframe(volumes.tail(10))
    
    else:
        st.error("Unable to fetch stock data. Please check your internet connection and try again.")

else:
    st.info("👈 Please select stocks and date range from the sidebar to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
**📝 Project Notes:**
- Data source: Yahoo Finance API
- Dashboard built with Streamlit & Plotly
- Perfect for financial analysis and investment research
- **For GCP deployment:** Use `gcloud app deploy` with app.yaml
""")

# GCP Deployment files (separate files needed)
st.markdown("""
### 🚀 GCP Deployment Files Needed:

**requirements.txt:**
```
streamlit==1.28.0
yfinance==0.2.18
pandas==2.0.3
numpy==1.24.3
plotly==5.15.0
seaborn==0.12.2
matplotlib==3.7.2
```

**app.yaml:**
```yaml
runtime: python39
env: standard
instance_class: F2

automatic_scaling:
  min_instances: 1
  max_instances: 3

env_variables:
  PORT: 8080
```

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD streamlit run dashboard.py --server.port 8080 --server.address 0.0.0.0
```
""")
