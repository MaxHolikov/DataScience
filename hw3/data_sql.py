import pandas as pd
import sqlite3
import logger as lg
class SQLData:
    def __init__(self, data: pd.DataFrame):
      self.data = data
      self.db_name = 'data_robotics.db'

    def sql_connect(self):
     
      conn = sqlite3.connect(self.db_name)
      cursor = conn.cursor()
      # Создание таблицы 
      robotics_table=('''CREATE TABLE IF NOT EXISTS users (
      Year INTEGER,
      Indastry TEXT UNIQUE NOT NULL,
      Robots_Adopted INTEGER UNIQUE NOT NULL,
      Productivity_Gain FLOAT UNIQUE NOT NULL, 
      Cost_Savings FLOAT UNIQUE NOT NULL,
      Jobs_Displaced INTEGER UNIQUE NOT NULL,
      Training_Hours INTEGER UNIQUE NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )''')
      cursor.execute(robotics_table)
      
      self.data.to_sql('robotics_data', conn, if_exists='replace', index=False)
      # Закрываем соединение
      conn.commit()
      conn.close() 
           
    # Метод для просмотра данных (без параметров)
    def sql_check(self):
        conn = sqlite3.connect(self.db_name)
        
        # Получаем все данные
        df = pd.read_sql_query("SELECT * FROM robotics_data", conn)
        
        print(f"\nВсего записей в БД: {len(df)}")
        print("\nПервые 5 записей:")
        for idx, row in df.head(5).iterrows():
            print(f"Год: {row['Year']}, Отрасль: {row['Industry']}, "
                  f"Кол-во роботов: {row['Robots_Adopted']}, "
                  f"Производительность: {row['Productivity_Gain']}%, "
                  f"Экономия: {row['Cost_Savings']} млн $, "
                  f"Сокращение персонала: {row['Jobs_Displaced']}, "
                  f"Обучение: {row['Training_Hours']} ч")
    
        conn.close()
        return df

    # Метод для выполнения произвольных SQL запросов
    def sql_query(self, query):
        conn = sqlite3.connect(self.db_name)
        try:
          result = pd.read_sql_query(query, conn)
          if len(result) > 0:
            print("\n   Результат запроса:")
            print(result)
          else:
            print("   Нет данных за указанный период")
            lg.log_event("INFO", f"SQL query executed. Found {len(result)} records")
        except Exception as e:
          lg.log_event("ERROR", f"SQL query error: {str(e)}")
          print(f" ОШИБКА при выполнении SQL запроса: {str(e)}")
        
        conn.close()
        return result

