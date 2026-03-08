import logger as lg
import pandas as pd
import numpy as np

def get_outlier_bounds(data,col,q_low,q_up):
    '''Функция для вычисления границ выбросов методом межквартильного размаха (IQR)
    Параметры:
    data - DataFrame с данными
    col - название колонки для анализа
    q_low - нижний квантиль (например, 0.25 для Q1)
    q_up - верхний квантиль (например, 0.75 для Q3)
    '''
    Q1=data[col].quantile(q_low)
    Q3=data[col].quantile(q_up)
    iqr=Q3-Q1
    up_limit=Q3+1.5*iqr
    low_limit=Q1-1.5*iqr
    return low_limit,up_limit

def get_outliers_percent(data, col, q_low, q_up):     #Функция для вычисления процента выбросов в данных
    low, up = get_outlier_bounds(data, col,q_low, q_up)
    outliers = [elem for elem in data[col] if (elem > up) or (elem < low)]
    return len(outliers) / data[col].shape[0] * 100

def rewrite_outliers(data,col, q_low, q_up):          #Функция для замены выбросов на граничные значения
    low,up=get_outlier_bounds(data,col, q_low, q_up)
    data.loc[(data[col]<low),col]=low
    data.loc[(data[col]>up),col]=up
    return data

def remove_outliers(data,col, q_low, q_up):           #Функция для полного удаления строк, содержащих выбросы
    low,up=get_outlier_bounds(data,col, q_low, q_up)
    data = data[(data[col] <= up) & (data[col] >= low)]
    return data