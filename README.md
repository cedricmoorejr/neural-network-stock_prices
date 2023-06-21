# User Input
```
start_date = "2020-01-01"
end_date = "2023-06-19"
ticker = "FDX" 

# Define the sequence length (number of previous time steps to consider)
choose_sequence_length = 60

# Split the data into training and testing sets
choose_split_size = 0.8
```
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
# Step 2: Load and Preprocess the Data
```
# Load the stock price data into a Pandas DataFrame
from StockPrice import StockPrice as Stock
df = Stock.stock_to_df(ticker=ticker, start=start_date, end=end_date, columns=["Date", "Close"])
```
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
# Performs linear interpolation to estimate the missing values in the 'Close' column of the DataFrame
data_1 = df.copy()
data_1["Close"].interpolate(method="linear", inplace=True)
print(data_1)
```

While we're at it, lets look at a few more interpolation methods.
```
# Polynomial Interpolation
data_2 = df.copy()
data_2["Close"].interpolate(method="polynomial", order=2, inplace=True)
print(data_2)
```
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


