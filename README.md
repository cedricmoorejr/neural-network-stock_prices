# Data

```
model = Sequential()

# Add an LSTM layer with 50 units and input shape
model.add(LSTM(50, input_shape=(choose_sequence_length, 1), return_sequences=True))
model.add(Dropout(0.2))

# Add another LSTM layer with 50 units
model.add(LSTM(50))
model.add(Dropout(0.2))

# Add a dense layer with 1 unit to output the predicted stock price
model.add(Dense(1))

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")
```
