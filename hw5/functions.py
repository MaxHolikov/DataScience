import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logger as lg
import category_encoders as ce
# Для полиномиальной регрессии
from sklearn.preprocessing import PolynomialFeatures

# Для моделей машинного обучения
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Для метрик
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Для сохранения результатов
import csv
import os
from datetime import datetime

R2 = {}      # определяем глобальные словари!
RMSE = {}    # определяем глобальные словари!
MAE = {}     # определяем глобальные словари!


#энкодирование и разделение данных 
def encode_categorical(data, cols): 
    encoder = ce.BinaryEncoder(cols=cols)
    data_encoded = encoder.fit_transform(data)
    return data_encoded


def save_metrics_to_csv(name, r2, rmse, mae, filename='metrics.csv'):
    metrics_df = pd.DataFrame([{
        'Model': name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }])
    print(metrics_df['R2'])
    print('=====================')
    
    if os.path.exists(filename):
        metrics_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(filename, index=False)

def train_model_standard(model, x_train, y_train, name):
    # model training
    if name == 'PolynomialRegression':
        poly = PolynomialFeatures(degree=2)  # Указываем степень полинома
        x_train = poly.fit_transform(x_train)
    model.fit(x_train,y_train)

def predict(model, x_test, x_train, name):
    # values predicting
    if name == 'PolynomialRegression':
        poly = PolynomialFeatures(degree=2)  # Указываем степень полинома
        x_train = poly.fit_transform(x_train)
        x_test = poly.transform(x_test)
    pre = model.predict(x_test)
    return pre

def calculate_metrics(model, x_test, y_test, pre, name, sequential = False, filename='metrics.csv'):
    # Metrics calculation
    global R2, RMSE, MAE
    acc = r2_score(y_test, pre)
    rmse = np.sqrt(mean_squared_error(y_test, pre))
    mae = mean_absolute_error(y_test, pre)
    
    R2[name] = acc
    RMSE[name] = rmse
    MAE[name] = mae

    save_metrics_to_csv(name, acc, rmse, mae, filename)

    # Output for model
    lg.log_event("INFO",f'Model: {name}\n')
    lg.log_event("INFO",f'R²: {acc}, RMSE: {rmse}, MAE: {mae} \n ======================')
    print(f'Model: {name}')
    print(f'R²: {acc}, RMSE: {rmse}, MAE: {mae}')

    metrics_df = pd.DataFrame([{
        'Model': name,
        'R2': R2,
        'RMSE': rmse,
        'MAE': mae
    }])

    # Visualizations with plots
    plt.figure(figsize=(12, 6))

    # Actual vs Predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, pre)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')

    # Residuals diagram
    plt.subplot(1, 2, 2)
    sns.histplot(y_test - pre, bins=30, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()


def make_metric_plot(metric_data, name):

    names = list(metric_data.keys())
    values = list(metric_data[name].values)
    
    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color='skyblue')
    plt.title(name)
    plt.xlabel('Models')
    plt.ylabel(name)
    plt.xticks(rotation=45)
    bars = plt.bar(names, values, color='skyblue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')
    plt.show()

def compare_models_metrics(metric_data, save_plots=False):
    """
    Создает сравнительные графики для всех метрик моделей.
    
    Parameters:
    -----------
    metric_data : pandas.DataFrame
        DataFrame с колонками: Model, R2, RMSE, MAE
    save_plots : bool
        Сохранять ли графики в файлы
    """
    # Проверка наличия необходимых колонок
    required_cols = ['Model', 'R2', 'RMSE', 'MAE']
    for col in required_cols:
        if col not in metric_data.columns:
            lg.log_event("ERROR", f"Ошибка: Колонка '{col}' не найдена в данных!")
            lg.log_event("INFO","Доступные колонки: {metric_data.columns.tolist()}")
            return

    # Получаем данные
    models = metric_data['Model'].tolist()
    
    # Сокращаем названия моделей для лучшей читаемости
    short_names = []
    for model in models:
        if 'Linear' in model:
            short_names.append('Linear')
        elif 'Decision' in model:
            short_names.append('Decision Tree')
        elif 'Random' in model:
            short_names.append('Random Forest')
        elif 'Gradient' in model:
            short_names.append('Gradient Boost')
        elif 'Polynomial' in model:
            short_names.append('Polynomial')
        else:
            short_names.append(model[:15])
    
    # Цвета для графиков
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194']
    
    # Создаем фигуру с тремя подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение производительности моделей', fontsize=16, fontweight='bold')
    
    # 1. График MAE (средняя абсолютная ошибка)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(short_names, metric_data['MAE'], color=colors[:len(models)], 
                    edgecolor='black', linewidth=1)
    ax1.set_title('MAE (Средняя абсолютная ошибка)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Модели')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. График RMSE (корень из среднеквадратичной ошибки)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(short_names, metric_data['RMSE'], color=colors[:len(models)],
                    edgecolor='black', linewidth=1)
    ax2.set_title('RMSE (Корень из среднеквадратичной ошибки)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Модели')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. График R² (коэффициент детерминации)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(short_names, metric_data['R2'], color=colors[:len(models)],
                    edgecolor='black', linewidth=1)
    ax3.set_title('R² (Коэффициент детерминации)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Модели')
    ax3.set_ylabel('R²')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)  # R² обычно от 0 до 1
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Сравнительный график всех метрик (нормализованных)
    ax4 = axes[1, 1]
    
    # Нормализуем метрики для сравнения на одном графике
    mae_norm = (metric_data['MAE'] - metric_data['MAE'].min()) / (metric_data['MAE'].max() - metric_data['MAE'].min())
    rmse_norm = (metric_data['RMSE'] - metric_data['RMSE'].min()) / (metric_data['RMSE'].max() - metric_data['RMSE'].min())
    r2_norm = 1 - (metric_data['R2'] - metric_data['R2'].min()) / (metric_data['R2'].max() - metric_data['R2'].min())  # Инвертируем для сравнения
    
    x = np.arange(len(models))
    width = 0.25
    
    ax4.bar(x - width, mae_norm, width, label='MAE (норм.)', color='#FF6B6B', edgecolor='black')
    ax4.bar(x, rmse_norm, width, label='RMSE (норм.)', color='#4ECDC4', edgecolor='black')
    ax4.bar(x + width, r2_norm, width, label='R² (инверт.)', color='#45B7D1', edgecolor='black')
    
    ax4.set_title('Сравнение метрик (нормализованные)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Модели')
    ax4.set_ylabel('Нормализованное значение')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Сохранение графиков
    if save_plots:
        fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        lg.log_event("INFO","Графики сохранены в 'model_comparison.png'")
    
    # Вывод лучших моделей
    
    lg.log_event("INFO","ЛУЧШИЕ МОДЕЛИ ПО КАЖДОЙ МЕТРИКЕ")
    
    best_mae_idx = metric_data['MAE'].idxmin()
    best_rmse_idx = metric_data['RMSE'].idxmin()
    best_r2_idx = metric_data['R2'].idxmax()
    lg.log_event("INFO",f"Лучшая по MAE: {models[best_mae_idx]} - {metric_data['MAE'][best_mae_idx]:.2f}")
    lg.log_event("INFO",f"Лучшая по RMSE: {models[best_rmse_idx]} - {metric_data['RMSE'][best_rmse_idx]:.2f}")
    lg.log_event("INFO",f"Лучшая по R²: {models[best_r2_idx]} - {metric_data['R2'][best_r2_idx]:.3f}")
    
    return fig

