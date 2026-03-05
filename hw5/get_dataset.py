import os
import shutil
from google.colab import files
import zipfile
import pandas as pd
import logger as lg 


def load_dataset_pdf():
  # Определяем пути источника и назначения
  source_path = 'kaggle.json'
  destination_dir = os.path.expanduser('~/.kaggle')  
  destination_path = os.path.join(destination_dir, 'kaggle.json')
  # Создаем директорию назначения, если она не существует
  os.makedirs(destination_dir, exist_ok=True)
  try:
  # Перемещаем файл
    shutil.move(source_path, destination_path)
  except FileNotFoundError:
      lg.log_event("ERROR", "Cannot file: kaggle.json! Load file: kaggle.json!" ) 

  # Устанавливаем права доступа
  os.chmod(destination_path, 0o600)
  # Загружаем файл
  uploaded = files.upload()

def unpacking_zip(name_zip, name):
  print(name)
  with zipfile.ZipFile(name_zip, 'r') as zip_ref:
    zip_ref.extractall(name)


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
