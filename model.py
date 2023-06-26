## User Input
##-----------
start_date = "2020-01-01"
end_date = "2023-06-19"
ticker = "FDX" 

# Number of previous time steps to consider
# i.e., how many previous days' prices are used for prediction
choose_sequence_length = 60

# Split the data into training and testing sets
choose_split_size = 0.8







##===============================================================
##            Step 1: Import the required libraries            ==
##===============================================================
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

##================================================================
##             Step 2: Load and Preprocess the Data             ==
##================================================================
# Load the stock price data into a Pandas DataFrame
from StockPrice import StockPrice as Stock
df = Stock.stock_to_df(ticker=ticker, start=start_date, end=end_date, columns=["Date", "Close"])

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


## Performs linear interpolation to estimate the missing values in the 'Close' column of the DataFrame
data_1 = df.copy()
data_1["Close"].interpolate(method="linear", inplace=True)

## Normalization
norm = MinMaxScaler(feature_range=(0, 1)) # Normalize the closing prices to the range of 0 to 1
data_1["normalized_close_price"] = norm.fit_transform(data_1["Close"].values.reshape(-1, 1))


##===============================================================
##        Step 3: Prepare the Training and Testing Data        ==
##===============================================================
# Split data into sets
proportion = int(len(data_1) * choose_split_size) # proportion of the training set relative to the entire dataset
train_data = data_1[:proportion] # Split into Training Set
test_data = data_1[proportion:] # Split into Test Set

# Prepare the data in a format suitable for training an LSTM
from recurrent_neural_networks import sequence
# Create training sequences
Train_X, Train_Y = sequence.create(data = train_data["normalized_close_price"].values, length = choose_sequence_length)

# Create testing sequences
Test_X, Test_Y = sequence.create(data=test_data["normalized_close_price"].values, length = choose_sequence_length)

# This is the input data. It must be reshaped in the form of a 3D tensor.
Train_X = np.reshape(Train_X, (Train_X.shape[0], Train_X.shape[1], 1))
Test_X = np.reshape(Test_X, (Test_X.shape[0], Test_X.shape[1], 1))


##================================================================
##                 Step 4: Build the LSTM model                 ==
##================================================================
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


##================================================================
##                 Step 5: Train the LSTM model                 ==
##================================================================
# Training the model
model.fit(Train_X, Train_Y, epochs=50, batch_size=32, verbose=0)


##================================================================
##                Step 6: Evaluate Model Results                ==
##================================================================
# Evaluating the model on training and test data
Train_loss = model.evaluate(Train_X, Train_Y, verbose=0)
Test_loss = model.evaluate(Test_X, Test_Y, verbose=0)
print(f"Train Loss: {Train_loss:.4f}")  # Printing the training loss
print(f"Test Loss: {Test_loss:.4f}")  # Printing the test loss


##================================================================
##                   Step 7: Make predictions                   ==
##================================================================
# Making predictions on training and test data
Train_predictions = model.predict(Train_X)
Test_predictions = model.predict(Test_X)

# Inverse scale the predictions to reverse the scaling transformation 
# and bring the predictions and true labels back to their original scale. 
# This is done to compare the predicted and actual stock prices in their original form 
# for evaluation purposes.
inverted_Train_predictions = norm.inverse_transform(Train_predictions)
inverted_Test_predictions = norm.inverse_transform(Test_predictions)

##================================================================
##                   Step 8: Visualize                          ==
##================================================================
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



