import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
@st.cache_data
# Prepare dataset for training
@st.cache_data
def load_data():
    ticker_symbol = 'AAPL'

    # Fetch the data
    ticker_data = yf.Ticker(ticker_symbol)

    # Get all historical prices
    appleData = ticker_data.history("max")

    # Save to a CSV file
    appleData.to_csv("apple_stock_data.csv")

    # Read the CSV file into a DataFrame
    data = pd.read_csv('apple_stock_data.csv')

    data.columns = data.columns.str.strip().str.lower()  # Standardize column names to lowercase
    data['date'] = pd.to_datetime(data['date'], utc=True)  # Ensure 'date' is in datetime format
    data = data.sort_values(by='date')  # Sort by date
    return data

# Create necessary features
apple = load_data()
apple['day'] = apple['date'].dt.day
apple['month'] = apple['date'].dt.month
apple['year'] = apple['date'].dt.year
apple['Previous_open'] = apple['open'].shift(1)
apple['Previous_close'] = apple['close'].shift(1)
apple['Previous_high'] = apple['high'].shift(1)
apple['Previous_low'] = apple['low'].shift(1)
apple['Previous_volume'] = apple['volume'].shift(1)

# Drop the rows with missing values after shifting (first row will have NaN values)
apple = apple.dropna(subset=['Previous_open', 'Previous_close', 'Previous_high', 'Previous_low', 'Previous_volume'])

# Select features
X_close = apple[['open', 'Previous_open', 'Previous_high', 'Previous_low', 'Previous_volume', 'day', 'month', 'year']]
X_high = apple[['open', 'Previous_open', 'Previous_low', 'Previous_close', 'Previous_volume', 'day', 'month', 'year']]
X_low = apple[['open', 'Previous_open', 'Previous_high', 'Previous_close', 'Previous_volume','day', 'month', 'year']]
y_close = apple['close']
y_high = apple['high']
y_low = apple['low']
dates = apple['date']

# Scale features
scaler = StandardScaler()
X_close = scaler.fit_transform(X_close)
X_high = scaler.fit_transform(X_high)
X_low = scaler.fit_transform(X_low)

# Train-test split
X_train, X_test, y_high_train, y_high_test, dates_train, dates_test = train_test_split(X_high, y_high, dates, test_size=0.3, random_state=2022)
_, _, y_low_train, y_low_test, dates_train, dates_test = train_test_split(X_low, y_low, dates, test_size=0.3, random_state=2022)
_, _, y_close_train, y_close_test, dates_train, dates_test = train_test_split(X_close, y_close, dates, test_size=0.3, random_state=2022)

linear_high = LinearRegression()
linear_low = LinearRegression()
linear_close = LinearRegression()

# Train models
linear_high.fit(X_train, y_high_train)
linear_low.fit(X_train, y_low_train)
linear_close.fit(X_train, y_close_train)

# Predict on test data
predicted_high = linear_high.predict(X_test)
predicted_low = linear_low.predict(X_test)
predicted_close = linear_close.predict(X_test)

# Create a DataFrame for results
results = pd.DataFrame({
    'date': dates_test,
    'predicted_high': predicted_high,
    'actual_high': y_high_test.values,
    'predicted_low': predicted_low,
    'actual_low': y_low_test.values,
    'predicted_close': predicted_close,
    'actual_close': y_close_test.values
}).sort_values(by="date")

# Today's prediction
latest_data = apple.iloc[-1][['open', 'Previous_open', 'Previous_high', 'Previous_low', 'Previous_volume', 'day', 'month', 'year']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
TodayHigh = linear_high.predict(latest_data_scaled)[0]
TodayLow = linear_low.predict(latest_data_scaled)[0]
TodayClose = linear_close.predict(latest_data_scaled)[0]

# Streamlit Interface
st.title("Stock Price Prediction for AAPL (2014-2024 Testing Set)")

# Display today's predictions
st.write("### Today's Predicted Price")
st.metric(label="Predicted High", value=f"${TodayHigh:.2f}")
st.metric(label="Predicted Low", value=f"${TodayLow:.2f}")
st.metric(label="Predicted Close", value=f"${TodayClose:.2f}")

# Dropdown and input for date selection
st.write("### Search Stock Data by Date")
search_date = st.text_input("Enter a date (YYYY-MM-DD):", "")
selected_date = st.selectbox("Or select a date:", options=results['date'].dt.strftime('%Y-%m-%d'))
# Display results for the selected date
if search_date:
    try:
        search_date = pd.to_datetime(search_date)
        selected_row = results[results['date'] == search_date]
        if not selected_row.empty:
            st.write(f"### Results for {search_date.strftime('%Y-%m-%d')}")
            for _, row in selected_row.iterrows():
                st.write(f"**Predicted High:** ${row['predicted_high']:.2f}")
                st.write(f"**Actual High:** ${row['actual_high']:.2f}")
                st.write(f"**Predicted Low:** ${row['predicted_low']:.2f}")
                st.write(f"**Actual Low:** ${row['actual_low']:.2f}")
                st.write(f"**Predicted Close:** ${row['predicted_close']:.2f}")
                st.write(f"**Actual Close:** ${row['actual_close']:.2f}")
        else:
            st.error("No data found for the entered date.")
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
elif selected_date:
    selected_row = results[results['date'].dt.strftime('%Y-%m-%d') == selected_date]
    st.write(f"### Results for {selected_date}")
    for _, row in selected_row.iterrows():
        st.write(f"**Predicted High:** ${row['predicted_high']:.2f}")
        st.write(f"**Actual High:** ${row['actual_high']:.2f}")
        st.write(f"**Predicted Low:** ${row['predicted_low']:.2f}")
        st.write(f"**Actual Low:** ${row['actual_low']:.2f}")
        st.write(f"**Predicted Close:** ${row['predicted_close']:.2f}")
        st.write(f"**Actual Close:** ${row['actual_close']:.2f}")

# Line chart comparing predicted and actual close prices
st.write("### Comparison of Predicted vs. Actual Close Prices")
filtered_results = results[(results['date'] >= "2014-01-01") & (results['date'] <= "2024-12-31")]
st.line_chart(filtered_results.set_index('date')[['predicted_close', 'actual_close']])