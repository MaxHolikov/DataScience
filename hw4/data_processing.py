import pandas as pd
import logger as lg
import numpy as np

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
      return  pd.DataFrame({
        "Missing values": missing,
        "Percent (%)": percent
      })

    '''
    Удаление строк с пропущенными значениями.
        Параметры:
    - subset: Список колонок для проверки (по умолчанию все колонки)
    - threshold: Минимальное количество не-NaN значений для сохранения строки
                 (если 0 - удаляются строки с хотя бы одним NaN)
    - inplace: Если True - изменяет текущий DataFrame, если False - возвращает новый
    
    Возвращает:
    - DataFrame без пропущенных значений (если inplace=False)
    '''
    def drop_missing_rows(self, subset=None, threshold=0, inplace=False):
       
      rows_before = len(self.data)
    
    # Определяем колонки для проверки
      if subset is None:
        subset = self.data.columns
    
    # Удаляем строки
      if threshold > 0:
        # Удаляем строки, где количество не-NaN меньше порога
        clean_data = self.data.dropna(subset=subset, thresh=threshold)
      else:
        # Удаляем строки с любым NaN
        clean_data = self.data.dropna(subset=subset)
    
      rows_after = len(clean_data)
      rows_removed = rows_before - rows_after
    
      print(f"Удалено строк: {rows_removed}")
      print(f"Осталось строк: {rows_after}")
    
      if rows_removed > 0:
        print(f"Процент удаленных данных: {(rows_removed/rows_before)*100:.2f}%")
    
    # Логируем событие
      lg.log_event("INFO", f"Удалено {rows_removed} строк с пропущенными значениями")
    
      if inplace:
        self.data = clean_data
      else:
        return clean_data


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

    """
    Заполнение пропущенных значений с разделением по типам данных.
    Параметры:
    - numeric_strategy: Стратегия для числовых колонок ('mean', 'median', 'mode', 'constant')
    - categorical_strategy: Стратегия для категориальных колонок ('mode', 'constant')
    - custom_values: Словарь {колонка: значение} для конкретных значений
    - columns: Список колонок для обработки (по умолчанию все)
    """
    def fill_missing_new(self, numeric_strategy='mean', categorical_strategy='mode', custom_values=None, columns=None) -> None:

      if columns is None:
        columns = self.data.columns
    
    # Преобразуем custom_values в словарь, если передан
      if custom_values is None:
        custom_values = {}
    
      for col in columns:
        if col not in self.data.columns:
          lg.log_event("INFO", f"Предупреждение: колонка '{col}' не найдена")
          continue
            
        # Пропускаем если нет пропусков
        missing_count = self.data[col].isna().sum()
        if missing_count == 0:
          continue
        
        # Проверяем, есть ли специальное значение для этой колонки
        if col in custom_values:
          fill_value = custom_values[col]
          lg.log_event("INFO", f"Колонка '{col}': используем пользовательское значение '{fill_value}' ")
            
        else:
          # Определяем тип колонки
          is_numeric = pd.api.types.is_numeric_dtype(self.data[col])
          is_categorical = isinstance(self.data[col].dtype, pd.CategoricalDtype)
          is_object = self.data[col].dtype == 'object'
            
          # Выбираем стратегию в зависимости от типа
          if is_numeric:
            # Числовая колонка
            if numeric_strategy == 'mean':
              fill_value = self.data[col].mean()
            elif numeric_strategy == 'median':
              fill_value = self.data[col].median()
            elif numeric_strategy == 'mode':
              fill_value = self.data[col].mode()[0]
            else:
              fill_value = numeric_strategy  # конкретное число
                    
          else:
             # Категориальная или строковая колонка
            if categorical_strategy == 'mode':
              mode_values = self.data[col].mode()
              fill_value = mode_values[0] if len(mode_values) > 0 else "Unknown"
            elif categorical_strategy == 'constant':
              fill_value = "Unknown"  # или другое значение по умолчанию
            else:
              fill_value = categorical_strategy  # конкретная строка
        
        # Заполняем пропуски
        try:
            # Для категориальных колонок нужно добавить категорию если её нет
            if isinstance(self.data[col].dtype, pd.CategoricalDtype):
                if fill_value not in self.data[col].cat.categories:
                    self.data[col] = self.data[col].cat.add_categories([fill_value])
            
            self.data.fillna({col: fill_value}, inplace=True)
            lg.log_event("INFO", f"Колонка '{col}': заполнено '{missing_count}' пропусков -> '{fill_value}'")
            
        except Exception as e:
            lg.log_event("ERROR","Ошибка в колонке '{col}': {e}")


    """
    Универсальный метод для преобразования категориальных колонок в числовые
    Параметры:
    - df: DataFrame для преобразования
    - verbose: выводить информацию о преобразованиях
    
    Возвращает:
    - DataFrame с числовыми колонками
    - Словарь с информацией о преобразованиях
    """

    def convert_to_numeric(self, verbose=True):
      df=self.data
      df_numeric = df
      conversion_info = {}
    
    # Расширенные словари для преобразования
      binary_mappings = {
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        'present': 1, 'absent': 0,
        'male': 1, 'female': 0,
        'm': 1, 'f': 0,
        'positive': 1, 'negative': 0
      }
    
      ordinal_mappings = {
        'low': 1, 'medium': 2, 'high': 3,
        'mild': 1, 'moderate': 2, 'severe': 3,
        'small': 1, 'medium': 2, 'large': 3,
        'easy': 1, 'medium': 2, 'hard': 3,
        'rarely': 1, 'sometimes': 2, 'often': 3, 'always': 4
      }
    
      for col in df.columns:
        original_dtype = df[col].dtype
        
        # Пропускаем уже числовые колонки
        if pd.api.types.is_numeric_dtype(df[col]):
            if verbose:
                print(f"{col}: уже числовая")
            conversion_info[col] = {'type': 'numeric', 'original': original_dtype}
            continue
        
        # Получаем уникальные значения (без пропусков)
        unique_vals = df[col].dropna().unique()
        unique_lower = set(str(v).lower() for v in unique_vals)
        
        # 1. Проверка на бинарные значения
        if len(unique_lower) <= 2 and all(v in binary_mappings or v in ['yes', 'no'] for v in unique_lower):
            # Создаем маппинг для этой колонки
            col_mapping = {}
            for val in unique_vals:
                val_lower = str(val).lower()
                if val_lower in binary_mappings:
                    col_mapping[val] = binary_mappings[val_lower]
                elif val_lower == 'yes':
                    col_mapping[val] = 1
                elif val_lower == 'no':
                    col_mapping[val] = 0
                else:
                    col_mapping[val] = 0  # по умолчанию
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'binary',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: бинарная -> {col_mapping}")
        
        # 2. Проверка на порядковые значения
        elif any(v in ordinal_mappings for v in unique_lower):
            col_mapping = {}
            for val in unique_vals:
                val_lower = str(val).lower()
                if val_lower in ordinal_mappings:
                    col_mapping[val] = ordinal_mappings[val_lower]
                else:
                    # Если значение не найдено, используем среднее
                    col_mapping[val] = 2
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'ordinal',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: порядковая -> {col_mapping}")
        
        # 3. Обычные категориальные - используем Label Encoding
        else:
            # Сортируем уникальные значения для согласованности
            sorted_vals = sorted(unique_vals, key=lambda x: str(x))
            col_mapping = {val: i for i, val in enumerate(sorted_vals)}
            
            df_numeric[col] = df[col].map(col_mapping)
            conversion_info[col] = {
                'type': 'categorical',
                'mapping': col_mapping,
                'unique_original': list(unique_vals)
            }
            if verbose:
                print(f"{col}: категориальная -> {len(sorted_vals)} категорий")
        lg.log_event("INFO", f" Данные преобразованы к цифровым значениям")
        #print(df_numeric.head())
      return df_numeric

    