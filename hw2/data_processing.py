import pandas as pd

class DataProcessing:
    def __init__(self, data: pd.DataFrame):
      self.data = data

    #количество пропущенных значений
    def count_missing(self) -> pd.Series: # подсчет пропущенных значений
      return self.data.isna().sum().sum()
    
    #oтчет о пропущенных значениях
    def missing_report(self):
      missing = self.data.isnull().sum()
      print(f"Отчет пропущенных значений \n{missing} \n")
      percent = (missing / len(self.data)) * 100
      return pd.DataFrame({
        "Missing values": missing,
        "Percent (%)": percent
      })
   
    def fill_missing(self, strategy='mean', columns=None) -> None:
        '''
        Заполнение пропущенных значений.
        Параметры:
        - strategy: Стратегия заполнения ('mean', 'median', 'mode' или конкретное значение).
        - columns: Список столбцов для обработки (по умолчанию все числовые столбцы).
        '''
        if columns is None:
            # Выбираем только числовые столбцы
            columns = self.data.select_dtypes(include='number').columns

        for col in columns:
            if strategy == 'mean':
                fill_value = self.data[col].mean()
            elif strategy == 'median':
                fill_value = self.data[col].median()
            elif strategy == 'mode':
                fill_value = self.data[col].mode()[0]
            else:
                fill_value = strategy  # Конкретное значение

            self.data.fillna({col:fill_value}, inplace=True)


