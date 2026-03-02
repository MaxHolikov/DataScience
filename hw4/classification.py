import logger as lg
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

class Classification:
  def __init__(self, data: pd.DataFrame):
    self.data = data
    df=self.data

    #градиентный бустинг
  def grad_boosting(self, target_col='Heart Disease Status', test_size=0.25, random_state=3000):
    df=self.data
    # Проверка наличия целевой колонки
    if target_col not in self.data.columns:
      lg.log_event("ERROR",f"Ошибка: колонка '{target_col}' не найдена!")
      lg.log_event("INFO",f"Доступные колонки: {df.columns.tolist()}")
      return None
        
    # Разделяем на признаки и целевую переменную
    X = df.drop(target_col, axis=1)
    y = df[target_col]
        
    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Создание модели Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=random_state)
    
    # Обучение модели на обучающем наборе данных
    gb_classifier.fit(X_train, y_train)

    # Предсказание классов на тестовом наборе данных
    y_pred = gb_classifier.predict(X_test)

    # Вывод полного отчета
    report = classification_report(y_test, y_pred, zero_division=0)
    print("     Gradient Boosting:")
    # Точность
    accuracy = (y_pred == y_test).mean()
    print(f'    Accuracy: {accuracy:.2f}')
    print(report)
    print("=============================")

    # Логируем результат
    lg.log_event("INFO", f"Gradient Boosting завершен. Accuracy: {accuracy:.5f}")
    # Возвращаем результаты для возможного дальнейшего использования
    return {
            'model': gb_classifier,
            'report': report,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
    }

  def ada_boosting(self, target_col='Heart Disease Status', test_size=0.25, random_state=3000):
    df=self.data
    # Проверка наличия целевой колонки
    if target_col not in self.data.columns:
      lg.log_event("ERROR",f"Ошибка: колонка '{target_col}' не найдена!")
      lg.log_event("INFO",f"Доступные колонки: {df.columns.tolist()}")
      return None
        
    # Разделяем на признаки и целевую переменную
    X = df.drop(target_col, axis=1)
    y = df[target_col]
        
    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  
    # Создание и обучение классификатора AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada_classifier = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=200,
    learning_rate=0.1,
    random_state=3000
    )
    ada_classifier.fit(X_train, y_train)

    # Прогнозирование классов на тестовом наборе данных
    y_pred = ada_classifier.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print('     ADABoosting:')
    print(f'    Accuracy: {accuracy:.2f}')
    print(report)
    print("=============================")
    # Логируем результат
    lg.log_event("INFO", f"ADABoosting завершен. Accuracy: {accuracy:.5f}")
    # Возвращаем результаты для возможного дальнейшего использования
    return {
            'model': ada_classifier,
            'report': report,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
    }
  
  def extra_trees(self, target_col='Heart Disease Status', test_size=0.25, random_state=3000):
    df=self.data
    # Проверка наличия целевой колонки
    if target_col not in self.data.columns:
      lg.log_event("ERROR",f"Ошибка: колонка '{target_col}' не найдена!")
      lg.log_event("INFO",f"Доступные колонки: {df.columns.tolist()}")
      return None

    # Проверка распредел

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3000)

    # Создание и обучение классификатора Extra Trees
    clf = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', random_state=3000)
    clf.fit(X_train, y_train)

    # Прогнозирование и оценка точности
    y_pred = clf.predict(X_test)

    # Вывод метрик классификации
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    
    print('     ExtraTreesClassifier:')
    print(f'    Accuracy: {accuracy:.2f}')
    print(report)
    print("=============================")
    # Логируем результат
    lg.log_event("INFO", f"ExtraTreesClassifier завершен. Accuracy: {accuracy:.5f}")
    # Возвращаем результаты для возможного дальнейшего использования
    return {
            'model': clf,
            'report': report,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
    }
  

  def k_neighbors(self, target_col='Heart Disease Status', test_size=0.25, random_state=3000):
    df=self.data
    # Проверка наличия целевой колонки
    if target_col not in self.data.columns:
      lg.log_event("ERROR",f"Ошибка: колонка '{target_col}' не найдена!")
      lg.log_event("INFO",f"Доступные колонки: {df.columns.tolist()}")
      return None

    # Проверка распредел

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3000)
  
    # Создание и обучение модели K Neighbors
    knn = KNeighborsClassifier(n_neighbors=3)  # Задаем количество соседей (K=3)
    knn.fit(X_train, y_train)

    # Предсказание на тестовом наборе
    y_pred = knn.predict(X_test)

    # Вывод метрик классификации
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print('     K Neighbors Classifier:')
    print(f'    Accuracy: {accuracy:.2f}')
    print(report)
    print("=============================")
    # Логируем результат
    lg.log_event("INFO", f"K Neighbors Classifier завершен. Accuracy: {accuracy:.5f}")
    # Возвращаем результаты для возможного дальнейшего использования
    return {
            'model': knn,
            'report': report,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
    }

  def dec_tree(self, target_col='Heart Disease Status', test_size=0.25, random_state=3000):
    df=self.data
    # Проверка наличия целевой колонки
    if target_col not in self.data.columns:
      lg.log_event("ERROR",f"Ошибка: колонка '{target_col}' не найдена!")
      lg.log_event("INFO",f"Доступные колонки: {df.columns.tolist()}")
      return None

    # Проверка распредел

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3000)

    # Создание и обучение модели Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=3000)
    dt_classifier.fit(X_train, y_train)

    # Предсказание классов на тестовом наборе данных
    y_pred = dt_classifier.predict(X_test)

    # Вывод метрик классификации
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('     Decision Tree Classifier:')
    print(f'    Accuracy: {accuracy:.2f}')
    print(report)
    print("=============================")
    # Логируем результат
    lg.log_event("INFO", f"Decision Tree Classifier завершен. Accuracy: {accuracy:.5f}")
    # Возвращаем результаты для возможного дальнейшего использования
    return {
            'model': dt_classifier,
            'report': report,
            'accuracy': accuracy,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
    }
