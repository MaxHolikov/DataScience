import pandas as pd
import logger as lg
import numpy as np
from datetime import datetime
from sklearn.impute import KNNImputer

class DataProcessing:
    def __init__(self, data: pd.DataFrame):
      self.data = data

    #количество пропущенных значений
    def count_missing(self) -> pd.Series: # подсчет пропущенных значений
     # print(f"Количество пропущенных значений: {self.data.isna().sum().sum()}")
     # print('=================')
      return self.data.isna().sum().sum()
    
    def print_file(self):
    # Просматриваем первые несколько строк датасетаe
      print(self.data.head(5)) 
    # Получаем общую информацию о датасете
      print(self.data.info())
      print('=================')
      print(f"Уникальные значения:\n{self.data.nunique()}")       # Уникальные значения
      print('=================')

    #oтчет о пропущенных значениях
    def missing_report(self):
      missing = self.data.isnull().sum()
      print(f"Количество пропущенных значений: {self.data.isna().sum().sum()}")
      print('=================')
      print(f"Отчет пропущенных значений \n{missing}")
      print('=================')
      percent = (missing / len(self.data)) * 100
      return  pd.DataFrame({
        "Missing values": missing,
        "Percent (%)": percent
      })

    def fill_missing(self, strategy='mean', columns=None) -> None:
        """
        Заполнение пропущенных значений.
        Параметры:
        - strategy: Стратегия заполнения ('mean', 'median', 'mode' или конкретное значение).
        - columns: Список столбцов для обработки (по умолчанию все числовые столбцы).
        """
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

    def create_eda_report(self):
      """Создает EDA отчет с переносами строк"""
      report_parts = []
        # Общая информация
      report_parts.append(f"""
      ОБЩАЯ ИНФОРМАЦИЯ
      {'-'*30}
      Размер датасета: {self.data.shape[0]} строк × {self.data.shape[1]} колонок
      Пропущенных значений: {self.data.isnull().sum().sum()}
      Дубликатов: {self.data.duplicated().sum()}""")
      lg.log_event("INFO", f"Размер датасета:{self.data.shape[0]} строк x {self.data.shape[1]} колонок")
      lg.log_event("INFO", f"Пропущенных значений: {self.data.isnull().sum().sum()}")
      lg.log_event("INFO", f"Дубликатов: {self.data.duplicated().sum()}")
      # Типы данных
      dtypes_info = self.data.dtypes.value_counts()
      report_parts.append(f"""
      ТИПЫ ДАННЫХ
      {'-'*30}""")
      for dtype, count in dtypes_info.items():
        report_parts.append(f"      {dtype}: {count} колонок")
     # Статистика по каналам
      if 'Channel' in self.data.columns:
        channel_dist = self.data['Channel'].value_counts()
        report_parts.append(f"""
      РАСПРЕДЕЛЕНИЕ ПО КАНАЛАМ
      {'-'*30}
      Horeca (1): {channel_dist.get(1, 0)} клиентов
      Retail (2): {channel_dist.get(2, 0)} клиентов
      """)
      return "\n".join(report_parts)


    