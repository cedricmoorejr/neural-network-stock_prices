# Long Short Term Memory Stock Price Prediction

Table of Contents
-----------------
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [License](#license)

## Overview

This project demonstrates the use of Long Short-Term Memory (LSTM) models for stock price prediction. LSTM models are a type of recurrent neural network (RNN) that can learn patterns and dependencies in sequential data. In this project, the LSTM model is trained on historical stock price data and used to make predictions on unseen data.

## Features

- Retrieve historical stock data for a given ticker symbol
- Preprocess the data by handling missing values and scaling
- Create sequences from the data for LSTM model training
- Train the LSTM model and make predictions on unseen data
- Evaluate the model's performance and visualize the results

## Getting Started

### Installation

To get started with the project, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/cedricmoorejr/neural-network-stock_prices.git
```

2. Navigate to the project directory:

```shell
cd neural-network-stock_prices
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

### Configuration

Before running the code, you may need to configure the parameters to suit your needs. Open the `model.py` file and modify the following variables:

- `start_date`: The start date for retrieving historical stock data
- `end_date`: The end date for retrieving historical stock data
- `ticker`: The ticker symbol of the stock
- `choose_sequence_length`: The number of previous time steps to consider for prediction
- `choose_split_size`: The proportion of the training set relative to the entire dataset

## Usage

To use this project, follow these steps:

1. Ensure that you have configured the parameters in the `model.py` file as described in the [Configuration](#configuration) section.

2. Run the code:

```shell
python model.py
```

3. Explore the evaluation results and visualizations to assess the model's performance.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- TensorFlow
- Keras
- matplotlib
- yfinance
- statsmodels

Please make sure to install these dependencies before running the code.

## Contact

For any questions or suggestions, please feel free to reach out to the project maintainers:

- Cedric Moore Jr. - cedricmoorejunior@outlook.com

## License

This project is licensed under the [MIT License](LICENSE.txt).
