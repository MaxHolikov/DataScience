import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_all(df, save_path=None):
    """
    Строит гистограммы для всех признаков датасета Wholesale Customers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с данными
    save_path : str, optional
        Путь для сохранения изображения
    """
    
    # Список признаков для визуализации
    features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
                'Detergents_Paper', 'Delicassen']
    
    # Создаем сетку графиков 3x3 (так как 8 признаков)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Определяем тип признака и строим соответствующую гистограмму
        if feature in ['Channel', 'Region']:
            # Категориальные признаки
            value_counts = df[feature].value_counts().sort_index()
            
            # Задаем названия для каналов
            if feature == 'Channel':
                labels = ['Horeca' if x == 1 else 'Retail' for x in value_counts.index]
                colors = ['skyblue', 'lightcoral']
                title = 'Распределение по каналам продаж'
            else:  # Region
                labels = [f'Регион {int(x)}' for x in value_counts.index]
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
                title = 'Распределение по регионам'
            
            # Столбчатая диаграмма для категориальных признаков
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color=colors, edgecolor='black', alpha=0.7)
            
            # Подписи значений на столбцах
            for bar, count in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{count}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel('Количество клиентов')
            
        else:
            # Числовые признаки - строим гистограмму
            n, bins, patches = ax.hist(df[feature], bins=30, edgecolor='black', 
                                       color='steelblue', alpha=0.7)
            
            # Добавляем статистику
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Среднее: {mean_val:.0f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Медиана: {median_val:.0f}')
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Частота')
            ax.legend()
            
            title = f'Распределение признака {feature}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Скрываем последний (9-й) график
    axes[8].set_visible(False)
    plt.suptitle('Гистограммы признаков датасета Wholesale Customers', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    plt.show()


def plot_detailed_histograms(df):
    """
    Строит детальные гистограммы с разбивкой по каналам
    """
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Разделяем по каналам
        horeca_data = df[df['Channel'] == 1][feature]
        retail_data = df[df['Channel'] == 2][feature]
        
        # Строим гистограммы для каждого канала
        ax.hist(horeca_data, bins=30, alpha=0.5, label='Horeca', 
                color='skyblue', edgecolor='black')
        ax.hist(retail_data, bins=30, alpha=0.5, label='Retail', 
                color='lightcoral', edgecolor='black')
        
        # Статистика
        ax.axvline(horeca_data.mean(), color='blue', linestyle='--', 
                  linewidth=2, label=f'Horeca ср.: {horeca_data.mean():.0f}')
        ax.axvline(retail_data.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Retail ср.: {retail_data.mean():.0f}')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Частота')
        ax.set_title(f'Распределение {feature} по каналам')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Детальный анализ распределений по каналам продаж', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_log_histograms(df):
    """
    Строит гистограммы с логарифмической шкалой для лучшей визуализации
    """
    
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Логарифмическая шкала
        ax.hist(np.log1p(df[feature]), bins=30, edgecolor='black', 
                color='steelblue', alpha=0.7)
        
        # Статистика в логарифмической шкале
        mean_log = np.log1p(df[feature]).mean()
        median_log = np.log1p(df[feature]).median()
        
        ax.axvline(mean_log, color='red', linestyle='--', linewidth=2,
                  label=f'Среднее (log): {mean_log:.2f}')
        ax.axvline(median_log, color='green', linestyle='--', linewidth=2,
                  label=f'Медиана (log): {median_log:.2f}')
        
        ax.set_xlabel(f'log({feature} + 1)')
        ax.set_ylabel('Частота')
        ax.set_title(f'Логарифмическое распределение {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Гистограммы с логарифмической шкалой', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def print_histogram_statistics(df):
    """
    Выводит статистику по каждому признаку
    """
    
    features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
                'Detergents_Paper', 'Delicassen']
    
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ПО ПРИЗНАКАМ")
    print("=" * 50)
    
    for feature in features:
        print(f"\n {feature}:")
        
        if feature in ['Channel', 'Region']:
            # Категориальные признаки
            counts = df[feature].value_counts().sort_index()
            for val, count in counts.items():
                if feature == 'Channel':
                    name = 'Horeca' if val == 1 else 'Retail'
                else:
                    name = f'Регион {val}'
                print(f"   {name}: {count} записей ({count/len(df)*100:.1f}%)")
        else:
            # Числовые признаки
            print(f"   Минимум: {df[feature].min():.0f}")
            print(f"   Максимум: {df[feature].max():.0f}")
            print(f"   Среднее: {df[feature].mean():.0f}")
            print(f"   Медиана: {df[feature].median():.0f}")
            print(f"   Стандартное отклонение: {df[feature].std():.0f}")
            
            # Квантили
            print(f"   25% квантиль: {df[feature].quantile(0.25):.0f}")
            print(f"   75% квантиль: {df[feature].quantile(0.75):.0f}")
            
            # Выбросы
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)).sum()
            print(f"   Выбросы: {outliers} ({outliers/len(df)*100:.1f}%)")

def plot_density_histograms(df):
    """
    Строит гистограммы плотности распределения
    """
    
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Гистограмма с плотностью
        sns.histplot(data=df, x=feature, hue='Channel', kde=True, 
                    palette={1: 'skyblue', 2: 'lightcoral'},
                    alpha=0.5, ax=ax)
        
        ax.set_title(f'Распределение {feature} (с плотностью)')
        ax.set_xlabel(feature)
        ax.set_ylabel('Плотность')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Гистограммы плотности распределения по каналам', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
