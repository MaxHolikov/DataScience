import os
import shutil
from google.colab import files
import zipfile
import pandas as pd
import logger as lg 

def view_file(name):
  #Показать файлы в директории
  print(os.listdir(name))

def read_file(name):
  df = pd.read_csv(name)

# Просматриваем первые несколько строк датасета
  print(df.head()) 

# Получаем общую информацию о датасете
  print(df.info())

# Выводим статистические данные о числовых столбцах
  #print(df.describe())
