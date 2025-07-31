import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start='2023-01-01', end='2025-05-31', save_path='data/{ticker}_data.csv'):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(save_path.format(ticker=ticker))
    return df

download_stock_data('AAPL')
