## Code Walkthrough

This walkthrough provides a step-by-step explanation of the code.

### Step 0: User Input

The user provides input parameters for the code, including the start date, end date, ticker, choose_sequence_length, and choose_split_size.
```
start_date = "2020-01-01"
end_date = "2023-06-19"
ticker = "FDX" 

# Number of previous time steps to consider
# i.e., how many previous days' prices are used for prediction
choose_sequence_length = 60

# Split the data into training and testing sets
choose_split_size = 0.8
```
### Step 1: Import the Required Libraries

The necessary libraries, including `numpy`, `pandas`, `sklearn`, and `tensorflow.keras`, are imported.
# Step 1: Import the Required Libraries 
```
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```


### Step 2: Load and Preprocess the Data
Import the custom function in the 'StockPrice.py' file and supply information to function. This defines a class called StockPrice that provides functionality to retrieve historical stock data for a given ticker symbol. 
```
from StockPrice import StockPrice as Stock   # Custom class
df = Stock.stock_to_df(ticker=ticker, start=start_date, end=end_date, columns=["Date", "Close"])
```
![image](https://github.com/cedricmoorejr/neural-network-stock_prices/assets/136417849/75e395e8-7dbe-45b5-822e-a3fedccfaf31)

When dealing with stock price data that has gaps due to holidays and
weekends, we can encounter issues in the model such as disrupted patterns,
increased noise, and inaccurate volatility estimation. We could forward
fill the data, which is essentially assuming that the stock price remains
constant from the last observed value until the next available value.
While this is not always accurate, it provides a reasonable approximation
for short gaps like weekends and holidays. However, in this example, we
want to be as accurate as possible, so we are going to look at some
alternative approaches. So lets prep and clean the data.


```
# Find missing dates
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Get the minimum and maximum dates from the DataFrame
startDate = df.index.min()
endDate = df.index.max()

# Create a complete date range from the minimum to the maximum date with daily frequency
complete_date_range = pd.date_range(start=startDate, end=endDate, freq="D")

# Identify the missing dates by comparing the complete date range with the existing dates in the DataFrame
missing_dates = complete_date_range[~complete_date_range.isin(df.index)]

# Add the missing dates to the DataFrame with Close prices as NaN
df = df.reindex(complete_date_range).reset_index(drop=False).rename(columns={"index": "Date"})
del startDate, endDate, complete_date_range
```

One of those approaches is by using interpolation
methods. Since this is time series data, we will use linear interpolation.
This method will estimate the missing values based on the available data points.

```
# Performs Linear Interpolation to estimate the missing values in the 'Close' column of the DataFrame
data_1 = df.copy()
data_1["Close"].interpolate(method="linear", inplace=True)
print(data_1)
```

While we're at it, lets look at a few more interpolation methods. This method is called Polynomial Interpolation.
A polynomial of degree "n" has "n + 1" coefficients and can approximate a
curve of degree "n". By increasing the order, you increase the flexibility
of the polynomial to fit the data more closely, but there is also a higher
risk of overfitting or introducing noise.  In the case of polynomial
interpolation, increasing the order can capture more intricate patterns in
the data but may also result in a more erratic fit, especially when
dealing with noisy or limited data points. Conversely, a lower order may
yield a smoother but less accurate approximation.  It's important to
strike a balance when choosing the order. A good practice is to start with
a low order (such as 1 or 2) and gradually increase it if needed,
carefully assessing the trade-off between accuracy and complexity.  For
example, using order=2 in polynomial interpolation means fitting a
quadratic polynomial to the data points. This polynomial will have three
coefficients (a, b, c), and the interpolated values will be obtained by
evaluating the quadratic equation at the missing positions.

```
# Polynomial Interpolation
data_2 = df.copy()
data_2["Close"].interpolate(method="polynomial", order=2, inplace=True)
print(data_2)
```



This last interpolation method is called Spline Interpolation.
A spline is a piecewise-defined function that consists of multiple
polynomial segments, where each segment is determined by a set of control
points. The order of the spline determines the degree of the polynomials
used within each segment.  When performing spline interpolation with
method='spline', increasing the order allows for higher degree polynomials
within each segment, resulting in a more flexible curve that can closely
fit the data points. Higher-order splines can capture more complex
patterns and exhibit more local variations.  It's important to note that
the "order" parameter in spline interpolation does not refer to the
overall degree of the polynomial curve but rather the degree of the
polynomials within each segment. The overall degree of the spline curve
will depend on the number of segments and the degree of the polynomials
used in each segment.  In the example code provided, using order=2 for
spline interpolation means fitting quadratic polynomials within each
segment of the spline. Quadratic splines have a continuous first
derivative, resulting in smooth curves. You can adjust the order parameter
to use higher values (e.g., 3 for cubic splines) to capture more intricate
patterns or lower values (e.g., 1 for linear splines) for smoother
approximations.

```
# Spline Interpolation
data_3 = df.copy()
data_3["Close"].interpolate(method="spline", order=2, inplace=True)
print(data_3)
```


The next approach we will try is by using advanced time-series modeling
techniques. One popular technique is the use of SARIMA (Seasonal
Autoregressive Integrated Moving Average) models. SARIMA models can handle
seasonality, trends, and autoregressive properties in the data.

```
# Seasonal Autoregressive Integrated Moving Average (SARIMA)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create a time index for the DataFrame
data_4 = df.copy()
data_4.set_index("Date", inplace=True)

# Fit a SARIMA model to the data
SARIMA_model = SARIMAX(data_4["Close"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fitted_SARIMA_model = SARIMA_model.fit()

# Predict missing values using the fitted model
missing_values = fitted_SARIMA_model.predict(start=missing_dates[0], end=missing_dates[-1])

# Fill in the missing values in the DataFrame
data_4.loc[missing_dates, "Close"] = missing_values
print(data_4)
```



As a final step in the preprocessing phase, we will begin scaling and normalizing the data. Scaling and normalizing data are preprocessing techniques used to
transform the values of a dataset to a specific range or distribution.
These techniques are commonly applied to improve the performance and
convergence of machine learning models, especially when working with
features that have different scales or distributions.
```
# Normalization
norm = MinMaxScaler(feature_range=(0, 1)) # Normalize the closing prices to the range of 0 to 1
data_1["normalized_close_price"] = norm.fit_transform(data_1["Close"].values.reshape(-1, 1))
```






### Step 3: Prepare the Training and Testing Data
The first thing that we are going to do in this step is divide the data into two sets: a training set and a test set. The
training set is used to train the LSTM model, while the test set is used
to evaluate the model's performance on unseen data.
```
# Split data into sets
proportion = int(len(data_1) * choose_split_size) # proportion of the training set relative to the entire dataset
train_data = data_1[:proportion] # Split into Training Set
test_data = data_1[proportion:] # Split into Test Set
```

Next we will prepare the data in such a way that its suitable for training an LSTM. To do that, I created a class called sequence.
LSTM models are a type of recurrent neural network (RNN) that are capable of learning patterns and dependencies in sequential data. To train an LSTM model, the input data needs to be structured as sequences, where each sequence represents a pattern of input features over a certain time window.

By creating sequences from the original data, the LSTM model can learn the temporal dependencies and patterns in the data. Each input sequence (X) corresponds to a set of previous data points, and the target value (y) is the next data point following that sequence. In this way, the model can learn to predict the next data point based on the preceding sequence.

For time-series data like stock prices, it's important to create sequences of input-output pairs for the LSTM model. Each input sequence contains a window of previous days' prices, and the corresponding output is the price of the next day. The sequence length is a parameter that determines how many previous days' prices are used for prediction.

```
# Prepare the data in a format suitable for training an LSTM
from recurrent_neural_networks import sequence
# Create training sequences
Train_X, Train_Y = sequence.create(data = train_data["normalized_close_price"].values, length = choose_sequence_length)

# Create testing sequences
Test_X, Test_Y = sequence.create(data=test_data["normalized_close_price"].values, length = choose_sequence_length)

# This is the input data. It must be reshaped in the form of a 3D tensor.
Train_X = np.reshape(Train_X, (Train_X.shape[0], Train_X.shape[1], 1))
Test_X = np.reshape(Test_X, (Test_X.shape[0], Test_X.shape[1], 1))
```





### Step 4: Build the LSTM Model
Next we will focus on building the model.
```
# Begin defining model architecture
model = Sequential()

# Add an LSTM layer with 50 units and specify the input shape
# choose_sequence_length is the length of each sequence
# The input shape is (choose_sequence_length, 1) since we have 1 feature
# Set return_sequences=True to return the output sequence rather than just the last output
model.add(LSTM(50, input_shape=(choose_sequence_length, 1), return_sequences=True))

# Add a dropout layer with a rate of 0.2
# Dropout is a regularization technique to prevent overfitting
# It randomly sets a fraction of input units to 0 during training
model.add(Dropout(0.2))

# Add another LSTM layer with 50 units
# Since return_sequences is not specified, it defaults to False
# So, this LSTM layer will only return the last output of the sequence
model.add(LSTM(50))

# Add another dropout layer
model.add(Dropout(0.2))

# Add a Dense layer with 1 unit
# This layer is the output layer of the model
model.add(Dense(1))

# Compile the model
# Use "mean_squared_error" as the loss function since it's a regression problem
# Use the "adam" optimizer, which is a popular choice for gradient-based optimization
model.compile(loss="mean_squared_error", optimizer="adam")
```
In this step, we are essentially doing a number of things:
- Defining the model architecture: The code block outlines the structure and layers of the LSTM model. It sets up the sequence of layers in a sequential manner, starting with LSTM layers, followed by dropout layers, and ending with a dense output layer.

- LSTM layer configuration: The LSTM layers are the core components of the model. They are responsible for learning and capturing temporal patterns in sequential data. In the code block, two LSTM layers are added to the model. The first LSTM layer has 50 units, specified using LSTM(50). The input shape is set to (choose_sequence_length, 1) to match the input data. By setting return_sequences=True, the layer returns the entire output sequence instead of just the last output, allowing for deeper learning.

- Dropout regularization: Dropout layers, added after each LSTM layer, help prevent overfitting. They randomly set a fraction of input units to 0 during training, forcing the model to learn more robust and generalized representations. In the code block, dropout layers with a dropout rate of 0.2 are added using Dropout(0.2).

- Output layer configuration: The final dense layer, added with Dense(1), serves as the output layer of the model. It consists of a single unit, which is appropriate for a regression problem where the goal is to predict a continuous numerical value.

- Model compilation: After defining the architecture, the model is compiled using model.compile(). The loss function is set to "mean_squared_error", which measures the mean squared difference between the predicted and actual values. The optimizer "adam" is chosen for gradient-based optimization during training.

Overall, this code block helps configure the LSTM model by defining its architecture, including LSTM layers for sequence learning, dropout layers for regularization, and an output layer for prediction. It also compiles the model with an appropriate loss function and optimizer for the specific regression task at hand.


### Step 5: Train the LSTM Model
```
# Training the model
model.fit(Train_X, Train_Y, epochs=50, batch_size=32, verbose=0)
```
The code snippet `model.fit(Train_X, Train_Y, epochs=50, batch_size=32, verbose=0)` serves the purpose of training the LSTM model using the provided training data. Here's an explanation of each parameter in the `fit` method:

- `Train_X`: The input training data (features) used to train the LSTM model. It is a 3D tensor representing the input sequences.
- `Train_Y`: The target training data (labels) used to train the LSTM model. It represents the expected output for each input sequence.
- `epochs`: The number of times the training data will be iterated over during the training process. Each epoch consists of one pass through the entire training dataset.
- `batch_size`: The number of samples used in each iteration to update the model's weights. The dataset is divided into batches, and the model is updated after processing each batch.
- `verbose`: A parameter that controls the verbosity of the training process. Setting `verbose=0` means no progress updates will be printed during training.

During the training process, the model adjusts its internal weights based on the provided training data and tries to minimize the defined loss function (mean squared error in this case). The goal is to optimize the model's ability to make accurate predictions on the training data.

By executing the `fit` method, the LSTM model learns to capture patterns and dependencies in the training data, fine-tuning its parameters to improve its predictive performance. The specified number of epochs determines the number of iterations through the training data, while the batch size controls the granularity of weight updates.




### Step 6: Evaluate Model Results
After training the model, it is expected to generalize its learned patterns to unseen data and make predictions on new or test data. The effectiveness of the trained model can be assessed by evaluating its performance on separate test data or by inspecting metrics such as loss values during training.
```
# Evaluating the model on training and test data
Train_loss = model.evaluate(Train_X, Train_Y, verbose=0)
Test_loss = model.evaluate(Test_X, Test_Y, verbose=0)
print(f"Train Loss: {Train_loss:.4f}")  # Printing the training loss
print(f"Test Loss: {Test_loss:.4f}")  # Printing the test loss
```
By evaluating the model on both training and test data, you can assess its performance and gain insights into how well it has learned from the training data and generalized to unseen data. Monitoring the loss values helps determine if further adjustments or optimizations are needed to improve the model's performance.

![image](https://github.com/cedricmoorejr/neural-network-stock_prices/assets/136417849/4a2c2fed-b9e7-4891-906d-588bfa6cefac)

Based on the train loss and test loss values, it can be inferred that the LSTM model has achieved relatively low loss values during training and testing.

A low loss value indicates that the predicted values from the model are close to the actual values. In this case, the train loss is 0.0010 and the test loss is 0.0006. These values suggest that the model has been able to learn patterns in the training data and generalize well to the unseen test data.


### Step 7: Make Predictions
Now we move on to the final step. The code below is responsible for making predictions using the trained LSTM model on both the training and test datasets.
```
# Making predictions on training and test data
Train_predictions = model.predict(Train_X)
Test_predictions = model.predict(Test_X)

# Inverse scale the predictions to reverse the scaling transformation 
# and bring the predictions and true labels back to their original scale. 
# This is done to compare the predicted and actual stock prices in their original form 
# for evaluation purposes.
inverted_Train_predictions = norm.inverse_transform(Train_predictions)
inverted_Test_predictions = norm.inverse_transform(Test_predictions)
```
![image](https://github.com/cedricmoorejr/neural-network-stock_prices/assets/136417849/4693dad4-43de-4db9-9d7c-1a6a5b735c37)

By making predictions on both the training and test datasets, you can examine how well the LSTM model performs in capturing patterns and predicting values. Applying the inverse scaling transformation allows you to compare the predicted values on the original scale with the actual stock prices, enabling evaluation and analysis of the model's accuracy and effectiveness.


### Step 8: Visualization
After training the LSTM model, we can use it to make predictions on a separate test dataset. We can plot the actual values from the test dataset against the predicted values generated by the LSTM model. This visual comparison helps us understand how well the model is capturing the patterns and making accurate predictions.
```
import matplotlib.pyplot as plt

# Plot actual and predicted values on the training data with custom colors
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data["Close"], color="blue", label="Actual")
plt.plot(train_data.index[choose_sequence_length:], inverted_Train_predictions, color="orange", label="Predicted")
plt.title("Actual vs. Predicted Stock Prices (Training Data)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Plot actual and predicted values on the testing data with custom colors
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data["Close"], color="green", label="Actual")
plt.plot(test_data.index[choose_sequence_length:], inverted_Test_predictions, color="purple", label="Predicted")
plt.title("Actual vs. Predicted Stock Prices (Testing Data)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

![image](https://github.com/cedricmoorejr/neural-network-stock_prices/assets/136417849/795b2fa5-5346-4f9c-943f-66d1fe50e99c)


![image](https://github.com/cedricmoorejr/neural-network-stock_prices/assets/136417849/efe73455-b350-4d0f-8aca-e4d13ad9d467)


This concludes the step-by-step walkthrough of the code.

