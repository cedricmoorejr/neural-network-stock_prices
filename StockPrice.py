class StockPrice:
    """
    This class provides functionality to retrieve historical stock data for a given ticker symbol.
    """

    @staticmethod
    def stock_to_df(ticker, start, end, columns=None):
        """
        Retrieves historical stock data for the given ticker symbol within the specified date range.

        Args:
            ticker (str): Ticker symbol of the stock.
            start (str or datetime): Start date of the historical data range (format: YYYY-MM-DD).
            end (str or datetime): End date of the historical data range (format: YYYY-MM-DD).
            columns (list, optional): List of column names to include in the DataFrame. Default is None.

        Returns:
            pandas.DataFrame: DataFrame containing the historical stock data.
        """
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta

        # Validate date formats
        if isinstance(start, str):
            try:
                datetime.strptime(start, "%Y-%m-%d")  # Check if the start date has the correct format
            except ValueError:
                raise ValueError("Invalid start date format. Expected format: YYYY-MM-DD.")

        if isinstance(end, str):
            try:
                datetime.strptime(end, "%Y-%m-%d")  # Check if the end date has the correct format
            except ValueError:
                raise ValueError("Invalid end date format. Expected format: YYYY-MM-DD.")

        # Convert end date to datetime and add one day
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)  # Convert end date to datetime and add one day

        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d")  # Convert start date to datetime

        if start > end:
            raise ValueError("Start date is greater than end date.")  # Check if start is greater than end

        stock_data = yf.download(ticker, start, end).reset_index(drop=False)  # Download the stock data

        if columns is not None:
            valid_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]  # List of valid column names
            invalid_columns = [col for col in columns if col not in valid_columns]  # Find invalid column names
            if invalid_columns:
                available_columns = ", ".join(valid_columns)  # Join valid column names as a comma-separated string
                raise ValueError("Invalid column name(s): {}. Available columns: {}".format(", ".join(invalid_columns), available_columns))

            stock_data = stock_data[columns]  # Slice the DataFrame to include only the specified columns

        return stock_data


