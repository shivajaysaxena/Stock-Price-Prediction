# 📈 Google Stock Price Prediction using LSTM 🤖

## 🌟 Introduction

Welcome to our exciting journey of predicting Google stock prices using cutting-edge machine learning techniques! This document outlines the process of using Long Short-Term Memory (LSTM) networks, a powerful type of Recurrent Neural Network (RNN), to forecast stock prices. Buckle up as we dive into the world of financial time series analysis! 🚀

## 🧠 Key Concepts

### 1. 🔁 What is RNN (Recurrent Neural Network)?

![image](https://github.com/user-attachments/assets/70065643-92c9-42ca-ae46-5fe9cc29208e)

Recurrent Neural Networks (RNNs) are the time lords of the neural network world! They're designed to work with sequential data, making them perfect for tasks like stock price prediction.

Key features:
- 🎭 Process input sequences of any length
- 💾 Maintain an internal state (memory)
- 🔄 Share parameters across time steps

### 2. 📉 What is the Vanishing Gradient Problem?

The Vanishing Gradient Problem is the arch-nemesis of deep neural networks, especially RNNs. It's like trying to whisper a message through a long line of people - by the time it reaches the end, the message is lost!

![image](https://github.com/user-attachments/assets/e8bc10b5-7418-4f3b-b3d3-8bb25b4017f8)

Key points:
- 🔬 Gradients become extremely small during backpropagation
- 🐌 Earlier layers or time steps learn very slowly
- 🕰️ Particularly problematic for long-term dependencies
- 🧠 Makes it difficult for the network to learn from distant past

### 3. 🧬 What is LSTM (Long Short-Term Memory)?

Long Short-Term Memory (LSTM) networks are the superheroes that save us from the vanishing gradient problem! They're specially designed to capture long-term dependencies in sequential data.

![image](https://github.com/user-attachments/assets/44d89c74-f45b-4701-998a-0bde6e410537)

Key features:
- 🗃️ Introduce a memory cell for long-term information storage
- 🚪 Use gating mechanisms to control information flow
- 🧠 Can learn to store relevant information for long periods
- 📚 Effective for tasks requiring understanding of long-term context

## 🛠️ Implementation Details

### Importing the Training Set

```python
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
```

This code imports the training data from a CSV file containing Google stock prices from 2012 to 2016. It then extracts the 'Open' price values into a numpy array.

### Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

Feature scaling is applied using Min-Max normalization to scale the values between 0 and 1. This is crucial for neural networks to work effectively.

Normalization Formula:
```
X_normalized = (X - X_min) / (X_max - X_min)
```

Where:
- X is the original value
- X_min is the minimum value in the feature
- X_max is the maximum value in the feature

### Creating 60 Timestamps

```python
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
```

This code creates sequences of 60 timestamps for the LSTM model. Each input sequence (X_train) consists of 60 previous stock prices, and the corresponding output (y_train) is the next day's price. This structure allows the model to learn from the past 60 days to predict the next day's price.

### LSTM Model Architecture

The LSTM model is created using Keras with the following architecture:

1. First LSTM layer:
   ```python
   regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
   regressor.add(Dropout(0.2))
   ```
   - 50 neurons
   - Returns sequences for stacking LSTM layers
   - Input shape based on the training data
   - Dropout of 20% to prevent overfitting

2. Second LSTM layer:
   ```python
   regressor.add(LSTM(units = 50, return_sequences = True))
   regressor.add(Dropout(0.2))
   ```
   - 50 neurons
   - Returns sequences
   - 20% dropout

3. Third LSTM layer:
   ```python
   regressor.add(LSTM(units = 50, return_sequences = True))
   regressor.add(Dropout(0.2))
   ```
   - Similar to the second layer

4. Fourth LSTM layer:
   ```python
   regressor.add(LSTM(units = 50))
   regressor.add(Dropout(0.2))
   ```
   - 50 neurons
   - Does not return sequences (last LSTM layer)
   - 20% dropout

5. Output layer:
   ```python
   regressor.add(Dense(units = 1))
   ```
   - Dense layer with 1 unit for predicting the stock price

The model is compiled using the Adam optimizer and Mean Squared Error as the loss function, which is appropriate for regression problems.

## 📊 Stock Price Prediction

The model predicts stock prices for January 2017 using the following steps:

1. Prepare the test data by concatenating training and test datasets.
2. Create sequences of 60 timestamps for the test data.
3. Use the trained model to make predictions.
4. Inverse transform the predictions to get actual stock price values.
5. Plot the real and predicted stock prices for visual comparison.

The resulting plot shows:
- 🔴 Red line: Real Google Stock Price
- 🔵 Blue line: Predicted Google Stock Price

![download](https://github.com/user-attachments/assets/e9688547-e811-45ce-a527-c5f6b504933c)

The Model shows the general curve for how stocks will move not the sudden peaks and downs which is not possible to predict for now. The model preforms perfectly to see rise and decline in stock.

## 🎉 Conclusion

Congratulations! You've now mastered the art of predicting Google stock prices using LSTM networks. Remember, while this model provides valuable insights, always consider multiple factors when making investment decisions. Happy forecasting! 📈🚀
