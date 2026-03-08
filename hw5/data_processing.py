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
      print(f"Количество пропущенных значений: {self.data.isna().sum().sum()}")
      print('=================')
      return self.data.isna().sum().sum()
    
    def print_file(self):
    # Просматриваем первые несколько строк датасетаe
      print(self.data.head(4)) 
    # Получаем общую информацию о датасете
      print(self.data.info())
      print('=================')

    #oтчет о пропущенных значениях
    def missing_report(self):
      missing = self.data.isnull().sum()
      print(f"Отчет пропущенных значений \n{missing}")
      print('=================')
      percent = (missing / len(self.data)) * 100
      return  pd.DataFrame({
        "Missing values": missing,
        "Percent (%)": percent
      })

    # удаление названия модели из строки с маркой авто
    def del_model(self):
      self.data= self.data.drop(['Unnamed: 0', 'registration_date', 'power_kw', 'fuel_consumption_g_km', 'offer_description'], axis=1)
      self.data['model']=self.data['model'].apply(lambda x: ' '.join(x.split()[1:]))

    # Преобразует указанную колонку в числовой формат
    def col_to_numeric(self, col_name):
      self.data[col_name] = pd.to_numeric(self.data[col_name], errors='coerce')
      print(f"Колонка '{col_name}' преобразована в числовой формат")
    
    #Преобразует колонку с расходом топлива в числовой формат
    #Удаляет ' l/100 km' и заменяет запятые на точки
    def consumption_to_numeric(self, col_name):
      self.data[col_name] = self.data[col_name].str.replace(' l/100 km', '').str.replace(',', '.')
      self.data[col_name] = pd.to_numeric(self.data[col_name], errors='coerce')
      print(f"Колонка '{col_name}' с расходом топлива преобразована в числовой формат")

    #Удаляет пропущенные значения в заданных колонках
    def remove_null_values(self):
      cols = ['color',
              'year',
              'price_in_euro',
              'power_ps',
              'mileage_in_km']
      self.data=self.data.dropna(subset=cols)
      return self.data.dropna(subset=cols)

    def change_year(self):
      current_year = datetime.now().year
      self.data['age'] = current_year - self.data['year']
      self.data.drop(columns=['year'], inplace=True)

    '''
    Заполняет пропуски во всех числовых колонках с помощью KNN
    и добавляет заполненную колонку расхода в основной DataFrame.
    '''
    def impute_and_merge_consumption(self, n_neighbors=5):
   
    # Получаем все числовые колонки
      numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
    
      if 'fuel_consumption_l_100km' not in numeric_cols:
        # Если колонки нет, создаем её с NaN
        self.data['fuel_consumption_l_100km'] = np.nan
        numeric_cols.append('fuel_consumption_l_100km')
    
      #print(f"Числовые колонки для заполнения: {numeric_cols}")
    
    # Подготовка данных
      data_num = self.data[numeric_cols].copy()
    
    # Заполняем пропуски
      imputer = KNNImputer(n_neighbors=min(n_neighbors, len(data_num)-1))
      filled_array = imputer.fit_transform(data_num)
    
    # Создаем DataFrame с заполненными данными
      filled_data = pd.DataFrame(filled_array, columns=numeric_cols)
    
    # Сбрасываем индексы
      self.data.reset_index(drop=True, inplace=True)
      filled_data.reset_index(drop=True, inplace=True)
    
    # Обновляем колонку расхода в основном DataFrame
      self.data['fuel_consumption_l_100km'] = filled_data['fuel_consumption_l_100km']
    
    # Статистика
      missing_before = data_num['fuel_consumption_l_100km'].isna().sum()
      missing_after = self.data['fuel_consumption_l_100km'].isna().sum()
      lg.log_event("INFO", f"Пропусков в расходе ДО заполнения: {missing_before}" )
      lg.log_event("INFO", f"Пропусков в расходе ПОСЛЕ заполнения: {missing_after}" )

      return self

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

    