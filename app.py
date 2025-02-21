import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Page config
st.set_page_config(page_title="Google Stock Price Prediction", layout="wide")

def load_data():
    # Load training data
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values
    
    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    # Create sequences for training
    X_train = []
    y_train = []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, sc, training_set, dataset_train

def build_model(X_train):
    # Initialize the RNN
    regressor = Sequential()
    
    # Adding LSTM layers and Dropout regularization
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    
    # Output layer
    regressor.add(Dense(units=1))
    
    # Compile the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    return regressor

def make_predictions(model, sc):
    # Load test data
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    # Prepare input data for predictions
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return real_stock_price, predicted_stock_price

def plot_predictions(real_stock_price, predicted_stock_price):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(real_stock_price, color='red', label='Real Google Stock Price')
    ax.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
    ax.set_title('Google Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    return fig

# Main app
st.title('Google Stock Price Prediction')

# Sidebar
st.sidebar.header('Model Parameters')
epochs = st.sidebar.slider('Number of Epochs', 1, 200, 100)
batch_size = st.sidebar.slider('Batch Size', 16, 64, 32)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Model Training and Prediction')
    
    if st.button('Train Model and Make Predictions'):
        with st.spinner('Loading data...'):
            X_train, y_train, sc, training_set, dataset_train = load_data()
        
        with st.spinner('Building and training model...'):
            model = build_model(X_train)
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Train the model
            for epoch in range(epochs):
                model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                progress_text.text(f'Training Progress: {int(progress * 100)}%')
            
            progress_text.text('Training completed!')
        
        with st.spinner('Making predictions...'):
            real_stock_price, predicted_stock_price = make_predictions(model, sc)
            
            # Plot results
            st.subheader('Prediction Results')
            fig = plot_predictions(real_stock_price, predicted_stock_price)
            st.pyplot(fig)
            
            # Show prediction metrics
            mse = np.mean((real_stock_price - predicted_stock_price) ** 2)
            rmse = np.sqrt(mse)
            st.metric("Root Mean Square Error (RMSE)", f"${rmse:.2f}")

with col2:
    st.subheader('About')
    st.write("""
    This application uses a Long Short-Term Memory (LSTM) neural network to predict Google stock prices.
    
    The model is trained on historical stock data and can predict future stock prices based on the patterns it learns.
    
    **Features:**
    - Uses 60 time steps for sequence prediction
    - 4 LSTM layers with dropout regularization
    - Predicts next day's opening price
    
    **How to use:**
    1. Adjust the model parameters in the sidebar
    2. Click 'Train Model and Make Predictions'
    3. Wait for the training to complete
    4. View the predictions and performance metrics
    """)

st.sidebar.markdown("""
### Notes
- Higher number of epochs generally leads to better predictions but takes longer to train
- Batch size affects training speed and model convergence
- The model uses the previous 60 days of data to make predictions
""")
