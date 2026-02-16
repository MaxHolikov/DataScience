import pandas as pd
#import numpy as np
import requests

def load_data_csv(file_path):       #Загрузка данных из CSV файла
  try:
    df= pd.read_csv(file_path)
    print(f"Загружено {len(df)} строк из файла CSV \n")
    return pd.read_csv(file_path)
  except Exception as e:
    print(f"Ошибка при загрузке CSV: {e}. Укажите корректный путь к файлу CSV.\n")
    return None
  except FileNotFoundError:
    print(f"Файл {file_path}\n")
    return None

def load_data_json(file_path):   #Загрузка данных из JSON файла.
  try:
    df = pd.read_json(file_path)
    print(f"Загружено {len(df)} строк из файла JSON \n")
    return pd.read_json(file_path)
  except Exception as e:
    print(f"Ошибка при загрузке JSON: {e}. Укажите корректный путь к файлу JSON.\n")
    return None
  except FileNotFoundError:
    print(f"Файл {file_path}\n")
    return None

def load_data_api(url):  #Загрузка данных из API.
  params = {
    "api_key": "your_api_key",
    "param1": "value1",
    "param2": "value2"
  }
  try:
    response = requests.get(url, params=params)
    if response.status_code == 200:
      df = pd.DataFrame(response.json())
      print("Данные успешно загружены из API.")
    else:
      print(f"Ошибка при запросе к API: {response.status_code}")
  except Exception as e:
    print(f"Ошибка при загрузке из API: {e}")
  
  