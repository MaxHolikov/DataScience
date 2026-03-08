import pandas as pd
import logger as lg 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_histogram(data, numeric_col, main_title, x_title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[numeric_col], bins=20, kde=True, stat='probability')
    plt.title(main_title)
    plt.xlabel(x_title)
    plt.ylabel('Frequency')
    plt.show()

def create_set_histogram(data, cols):
    data[cols].hist(bins=10, figsize=(14, 10))
    plt.tight_layout()
    plt.show()

def create_categorical_diagram(data, col):
    plt.figure(figsize=(8, 10))
    # ИСПРАВЛЕНО: добавлен hue и legend=False
    sns.countplot(data=data, x=col, hue=col, palette='pastel', legend=False)
    plt.title('Количество по категориям')
    plt.xlabel('Категория')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def categorical_price_plot(data, col):
    plt.figure(figsize=(10, 8))
    # ИСПРАВЛЕНО: заголовки приведены в соответствие с содержимым графика
    sns.barplot(x=col, y='price_in_euro', data=data, hue=col, palette='viridis', legend=False)
    plt.title('Средние цены на автомобили по категориальной колонке')
    plt.xlabel(col)
    plt.ylabel('Средняя цена в евро')
    plt.xticks(rotation=45)  # Поворот меток по оси X для удобства чтения
    plt.show()
