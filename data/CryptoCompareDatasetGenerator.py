import pandas as pd
import numpy as np
import requests
from typing import Tuple
from sklearn.model_selection import train_test_split
from .CryptoCompareDataset import CryptoCompareDataset

API_SYMBOLS = [
  'btc',
  'eth',
  'xrp',
  'usdt',
  'usdc',
  'busd',
  'bnb',
  'tusd',
  'doge',
  'shib'
]
API_CURRENCIES = ['usd', 'eur']

def fetch_histoday_records(symbol: str, to: str, limit: int = 2000):
  """
  Fetch real-time data using the [Daily Pair OHLCV service](https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistoday):  
  - `time`: Timestamp of the specific day for which the data is provided.
  - `high`: Highest price of the CC during the specified day.
  - `low`: Lowest price of the CC during the specified day.
  - `open`: Opening price of the CC at the beginning of the specified day.
  - `volumefrom`: Trading volume of the CC, indicating the quantity of the CC traded "from" during the specified day.
  - `volumeto`: Total value or monetary volume of the CC traded "to" during the specified day. This value is usually measured in the currency in which the trading volume is denominated (e.g., USD).
  - `close`: Closing price of the CC at the end of the specified day.
  - `conversionType`: Type of conversion used for the price data. In this case, it is mentioned as "direct," indicating that the prices provided are direct market prices without any additional conversions or adjustments.
  - `conversionSymbol`: Currency code used for conversion, if any. For example, if the prices were converted to a different currency, the conversion symbol would specify the target currency code.
  """
  if symbol not in API_SYMBOLS:
    raise ValueError(f'Expected one of {API_SYMBOLS}, got {symbol} instead')
  if to not in API_CURRENCIES:
    raise ValueError(f'Expected one of {API_CURRENCIES}, got {to} instead')
  response = requests.get('https://min-api.cryptocompare.com/data/v2/histoday', {
    'fsym': symbol,
    'tsym': to,
    'limit': limit
  })
  return response.json().get('Data').get('Data')

class CryptoCompareDatasetGenerator:
  FEATURES = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
  TARGET = 'close'
  RANDOM_STATE = 42
  
  dataset: CryptoCompareDataset

  def __init__(self, symbol: str, to: str):
    """_summary_

    :param symbol: _description_
    :param to: _description_
    """
    records = fetch_histoday_records(symbol, to)
    df = pd.DataFrame.from_records(records)
    X, y = self._df_to_x_y(df)
    self.dataset = CryptoCompareDataset(symbol, to, X, y)

  def _df_to_x_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[self.FEATURES].to_numpy()
    y = df[self.TARGET].to_numpy()
    return X, y

  def train_valid_test_split(self, test_size = .2):
    """
    Split stored dataset.
    """
    
    # Fixed: Split the actual data (X and y), not the symbol and to strings
    X_train, X_test, y_train, y_test = train_test_split(
      self.dataset.x,  # Changed from self.dataset.symbol
      self.dataset.y,  # Changed from self.dataset.to
      test_size=test_size,
      random_state=self.RANDOM_STATE
    )
    
    X_train, X_valid, y_train, y_valid = train_test_split(
      X_train,
      y_train,
      test_size=.25,
      random_state=self.RANDOM_STATE
    )

    return (
      CryptoCompareDataset(self.dataset.symbol, self.dataset.to, X_train, y_train, 'train'),
      CryptoCompareDataset(self.dataset.symbol, self.dataset.to, X_valid, y_valid, 'test'),
      CryptoCompareDataset(self.dataset.symbol, self.dataset.to, X_test,  y_test, 'test')
    )

  def to_df(self) -> pd.DataFrame:
    """
    Convert stored dataset into a DataFrame.
    """
    df = pd.DataFrame({
      key: val
      for key, val in zip(self.FEATURES, zip(*self.dataset.x))
    })
    return df

if __name__ == '__main__':
  gen = CryptoCompareDatasetGenerator('btc', 'usd')