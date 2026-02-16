import pandas as pd

class DataProcessing:
    def __init__(self, data: pd.DataFrame):
      self.data = data

    def count_missing(self) -> pd.Series: # подсчет пропущенных значений
      return self.data.isna().sum()

    def missing_report(self):
      missing = self.data.isnull().sum()
      percent = (missing / len(self.data)) * 100
      return pd.DataFrame({
        "Missing values": missing,
        "Percent (%)": percent
      })
      
