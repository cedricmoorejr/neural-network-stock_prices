# Data

*User Input*
```
start_date = "2020-01-01"
end_date = "2023-06-19"
ticker = "FDX" 

# Define the sequence length (number of previous time steps to consider)
choose_sequence_length = 60

# Split the data into training and testing sets
choose_split_size = 0.8
```
Step 1: Import the required libraries 
```
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from StockPrice import StockPrice as Stock
from generate import sequences

```
