import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import matplotlib.pyplot as plt

# App Title
st.title("📈 Advanced Stock Price Prediction with LSTM")

# Plot predictions vs actual
def plot_prediction(test_data, predicted_data):
    plt.figure(figsize=(14, 6))
    plt.plot(test_data, color='blue', label='Actual Stock Price')
    plt.plot(predicted_data, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

# Sidebar for stock ticker input
st.sidebar.header("Stock Selection")
tickers = st.sidebar.multiselect(
    "Select Stock Tickers (e.g., AAPL, MSFT, TSLA, GOOG)",
    ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"]
)

# Date range input
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Load stock data - Modified to ensure data up to end_date
@st.cache_data
def load_data(ticker, start, end):
    try:
        end_buffer = end + timedelta(days=5)  # Buffer for yfinance
        data = yf.download(ticker, start=start, end=end_buffer)
        end_ts = pd.Timestamp(end)
        data = data[data.index <= end_ts]

        # Ensure the data goes up to the specified end date.
        if data.index[-1] < end_ts:
            st.warning(f"Data for {ticker} is incomplete up to {end}.  Last available date: {data.index[-1].strftime('%Y-%m-%d')}")
            # Consider using a shorter date range if the end date is crucial.
            # Or add some handling to fetch data from another source to fill the gap.
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Feature Engineering: Adding Moving Averages & Volume Change
def add_features(data):
    data["MA_10"] = data["Close"].rolling(window=10).mean()
    data["MA_50"] = data["Close"].rolling(window=50).mean()
    data["MA_200"] = data["Close"].rolling(window=200).mean()
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Return"] = data["Close"].pct_change()  # Use returns instead of raw prices
    data.dropna(inplace=True)  # Drop NaN values
    return data

# Improved LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(units=100, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50, activation="relu"),
        Dense(units=1)  # Predict the next return, not the raw price
    ])
    optimizer = Adam(learning_rate=0.001, decay=1e-6)  # Adam with decay
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# Create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Main app logic
if tickers:
    for stock_ticker in tickers:
        st.write(f"Fetching data for **{stock_ticker}** from {start_date} to {end_date}")
        stock_data = load_data(stock_ticker, start_date, end_date)
    
        if stock_data is not None and not stock_data.empty:
            st.write("Data Preview:")
            st.dataframe(stock_data.tail())

            # Apply feature engineering
            stock_data = add_features(stock_data)
            
            # Plot stock closing price
            st.write("Historical Stock Prices:")
            st.line_chart(stock_data['Close'])
        
            # Data Preprocessing
            st.header("Data Preprocessing for LSTM")
            data = stock_data[['Return']].values.reshape(-1,1)  # Use returns, not raw prices
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)

            # Prepare training and testing data
            st.write("Splitting data into training and testing sets...")
            training_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:training_size]
            test_data = scaled_data[training_size - 100:]  # Use last 100 days for prediction context

            time_steps = 100  # Increased from 60 to 100
            X_train, y_train = create_sequences(train_data, time_steps)
            X_test, y_test = create_sequences(test_data, time_steps)

            # Reshape input for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            st.write(f"Training data shape: {X_train.shape}")
            st.write(f"Testing data shape: {X_test.shape}")

            # Train LSTM Model
            st.write("### 🔄 Training the LSTM Model")
            if st.button(f"Train Model for {stock_ticker}"):
                model = build_lstm_model((X_train.shape[1], 1))
                early_stop = EarlyStopping(monitor='loss', patience=5)
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, callbacks=[early_stop])

                # Predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)  # Convert back to returns
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                predictions_actual = predictions.flatten()
                
                # Plot predictions vs actual values
                fig_pred = go.Figure()
                test_dates = stock_data.index[training_size + time_steps:]
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_test_actual.flatten(), mode="lines", name="Actual Price"))
                fig_pred.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), mode="lines", name="Predicted Price"))
                fig_pred.update_layout(title=f"{stock_ticker} Predictions vs Actual", xaxis_title="Date", yaxis_title="Stock Price")
                plot_prediction(y_test_actual, predictions)

                # Predict Next 7 Days - Modified for cleaner output
                st.write(f"📅 **Next 7-Day Forecast for {stock_ticker}**")



                # Get the last known closing price
                last_close_price = stock_data["Close"].iloc[-1]

                # Generate last 100 days of data for prediction
                last_100_days = scaled_data[-time_steps:].reshape((1, time_steps, 1))

                future_predictions = []
                predicted_prices = [last_close_price]  # Start with last known price

                for _ in range(7):
                    next_pred = model.predict(last_100_days)
                    predicted_return = scaler.inverse_transform(next_pred)[0, 0]  # Convert return to original scale

                    # Convert return to price using the last predicted price
                    next_price = predicted_prices[-1] * (1 + predicted_return)
                    predicted_prices.append(next_price)

                    # Update the sequence for the next prediction
                    last_100_days = np.append(last_100_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)



                # Display the next 7 days' predictions
                # Display the next 7 days' predictions -  Cleanest output
                st.write("### 📊 Predicted Prices for the Next 7 Days")


                # Create future dates - Corrected to start from the day AFTER the specified end date
                end_date_obj = pd.to_datetime(end_date)
                future_dates = [end_date_obj + timedelta(days=i) for i in range(1, 8)]

                # Create DataFrame for output - Corrected and CLEANED
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices[1:]})

                # CRITICAL: Convert Predicted Prices to NUMERIC (floats)
                forecast_df['Predicted Price'] = forecast_df['Predicted Price'].astype(float)

                # Convert dates to strings for display
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

                st.dataframe(forecast_df, hide_index=True)  # Hide index
                # Plot the predictions
                def plot_predictions(future_dates, future_predictions):
                    plt.figure(figsize=(14, 6))
                    plt.plot(future_dates, future_predictions, color='red', marker='o', linestyle='dashed', label='Predicted Stock Price')
                    plt.title('Stock Price Prediction for Next 7 Days')
                    plt.xlabel('Date')
                    plt.ylabel('Stock Price')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.grid()
                    st.pyplot(plt)

                plot_predictions(future_dates, predicted_prices[1:])


                # Metrics
                mse = mean_squared_error(y_test_actual, predictions_actual)
                mae = mean_absolute_error(y_test_actual, predictions_actual)
                r2 = r2_score(y_test_actual, predictions_actual)
                st.write(f"✅ **MSE**: {mse:.4f} | **MAE**: {mae:.4f} | **R²**: {r2:.4f}")


                csv = forecast_df.to_csv(index=True) #Include Index
                st.download_button(label="📥 Download Predictions", data=csv, file_name=f"{stock_ticker}_predictions.csv", mime="text/csv")
