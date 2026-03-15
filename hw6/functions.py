import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logger as lg


def compare_single_histogram(df_before, df_after, feature, bins=30):
    """
    Строит гистограммы для одного признака ДО и ПОСЛЕ рядом
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ДО
    axes[0].hist(df_before[feature], bins=bins, alpha=0.7, 
                 color='#1E88E5', edgecolor='black', linewidth=0.5)
    axes[0].axvline(df_before[feature].mean(), color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Среднее: {df_before[feature].mean():.0f}')
    axes[0].axvline(df_before[feature].median(), color='green', 
                    linestyle='--', linewidth=2,
                    label=f'Медиана: {df_before[feature].median():.0f}')
    axes[0].set_title(f'{feature} - ДО (n={len(df_before)})', fontweight='bold')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Частота')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ПОСЛЕ
    axes[1].hist(df_after[feature], bins=bins, alpha=0.7, 
                 color='#E53935', edgecolor='black', linewidth=0.5)
    axes[1].axvline(df_after[feature].mean(), color='red', 
                    linestyle='--', linewidth=2,
                    label=f'Среднее: {df_after[feature].mean():.0f}')
    axes[1].axvline(df_after[feature].median(), color='green', 
                    linestyle='--', linewidth=2,
                    label=f'Медиана: {df_after[feature].median():.0f}')
    axes[1].set_title(f'{feature} - ПОСЛЕ (n={len(df_after)})', fontweight='bold')
    axes[1].set_xlabel('Значение')
    axes[1].set_ylabel('Частота')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Сравнение распределения {feature}', y=1.05, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ====================== ГИСТОГРАММЫ С ПЛОТНОСТЬЮ ======================

def compare_density_histograms(df_before, df_after, features, bins=30):
    """
    Строит гистограммы с наложением плотности распределения
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Гистограмма ДО (прозрачная)
        ax.hist(df_before[feature], bins=bins, alpha=0.5, density=True,
                color='#1E88E5', edgecolor='black', linewidth=0.5,
                label=f'ДО (n={len(df_before)})')
        
        # Гистограмма ПОСЛЕ (прозрачная)
        ax.hist(df_after[feature], bins=bins, alpha=0.5, density=True,
                color='#E53935', edgecolor='black', linewidth=0.5,
                label=f'ПОСЛЕ (n={len(df_after)})')
        
        # Линии средних
        ax.axvline(df_before[feature].mean(), color='darkblue', 
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Ср. ДО: {df_before[feature].mean():.0f}')
        ax.axvline(df_after[feature].mean(), color='darkred', 
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Ср. ПОСЛЕ: {df_after[feature].mean():.0f}')
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Скрываем пустые подграфики
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Сравнение распределений ДО и ПОСЛЕ удаления выбросов (с плотностью)', 
                 y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ====================== ГИСТОГРАММЫ С ВЫДЕЛЕНИЕМ ВЫБРОСОВ ======================

def compare_with_outliers_highlight(df_before, df_after, features, bins=30):
    """
    Строит гистограммы с выделением областей, где были выбросы
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Вычисляем границы выбросов по IQR для исходных данных
        Q1 = df_before[feature].quantile(0.25)
        Q3 = df_before[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Гистограмма ДО
        n_before, bins_before, _ = ax.hist(df_before[feature], bins=bins, alpha=0.6,
                                            color='#1E88E5', edgecolor='black', linewidth=0.5,
                                            label=f'ДО (n={len(df_before)})')
        
        # Гистограмма ПОСЛЕ
        ax.hist(df_after[feature], bins=bins, alpha=0.6,
                color='#E53935', edgecolor='black', linewidth=0.5,
                label=f'ПОСЛЕ (n={len(df_after)})')
        
        # Выделяем зоны выбросов
        ax.axvspan(ax.get_xlim()[0], lower_bound, alpha=0.2, color='red', label='Зона выбросов')
        ax.axvspan(upper_bound, ax.get_xlim()[1], alpha=0.2, color='red')
        
        # Вертикальные линии границ
        ax.axvline(lower_bound, color='red', linestyle=':', linewidth=2, alpha=0.8)
        ax.axvline(upper_bound, color='red', linestyle=':', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Скрываем пустые подграфики
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Сравнение гистограмм с выделением зон выбросов', 
                 y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
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

def print_histogram_statistics_comparison(df_before, df_after):
    """
    Выводит статистику по каждому признаку в две колонки: ДО и ПОСЛЕ
    
    Parameters:
    -----------
    df_before : DataFrame
        Данные до удаления выбросов
    df_after : DataFrame
        Данные после удаления выбросов
    """
    
    features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
                'Detergents_Paper', 'Delicassen']
    
    # Заголовок с двумя колонками
    print("\n" + "=" * 110)
    print(f"{'--- Visualizations BEFORE Outlier Removal ---':^55}{'--- Visualizations AFTER Outlier Removal ---':^55}")
    print("=" * 110)
    print(f"{'Размер данных: ' + str(len(df_before)) + ' записей':^55}{'Размер данных: ' + str(len(df_after)) + ' записей':^55}")
    print("=" * 110)
    
    for feature in features:
        print("\n" + "-" * 110)
        
        # Заголовок признака в две колонки
        print(f"{feature.upper():^55}{feature.upper():^55}")
        print("-" * 110)
        
        if feature in ['Channel', 'Region']:
            # Категориальные признаки
            counts_before = df_before[feature].value_counts().sort_index()
            counts_after = df_after[feature].value_counts().sort_index()
            
            # Объединяем все возможные значения
            all_values = sorted(set(counts_before.index) | set(counts_after.index))
            
            for val in all_values:
                if feature == 'Channel':
                    name = 'Horeca' if val == 1 else 'Retail'
                else:
                    name = f'Регион {val}'
                
                # Данные ДО
                count_before = counts_before.get(val, 0)
                pct_before = count_before / len(df_before) * 100
                
                # Данные ПОСЛЕ
                count_after = counts_after.get(val, 0)
                pct_after = count_after / len(df_after) * 100
                
                # Вывод в две колонки
                before_str = f"{name}: {count_before} записей ({pct_before:.1f}%)"
                after_str = f"{name}: {count_after} записей ({pct_after:.1f}%)"
                print(f"{before_str:<55}{after_str:<55}")
        
        else:
            # Числовые признаки
            # Данные ДО (реальные из датафрейма)
            min_before = df_before[feature].min()
            max_before = df_before[feature].max()
            mean_before = df_before[feature].mean()
            median_before = df_before[feature].median()
            std_before = df_before[feature].std()
            q25_before = df_before[feature].quantile(0.25)
            q75_before = df_before[feature].quantile(0.75)
            
            # Выбросы по IQR
            Q1_before = df_before[feature].quantile(0.25)
            Q3_before = df_before[feature].quantile(0.75)
            IQR_before = Q3_before - Q1_before
            outliers_before = ((df_before[feature] < Q1_before - 1.5*IQR_before) | 
                              (df_before[feature] > Q3_before + 1.5*IQR_before)).sum()
            
            # Данные ПОСЛЕ (реальные из датафрейма)
            min_after = df_after[feature].min()
            max_after = df_after[feature].max()
            mean_after = df_after[feature].mean()
            median_after = df_after[feature].median()
            std_after = df_after[feature].std()
            q25_after = df_after[feature].quantile(0.25)
            q75_after = df_after[feature].quantile(0.75)
            
            # Выбросы после
            Q1_after = df_after[feature].quantile(0.25)
            Q3_after = df_after[feature].quantile(0.75)
            IQR_after = Q3_after - Q1_after
            outliers_after = ((df_after[feature] < Q1_after - 1.5*IQR_after) | 
                             (df_after[feature] > Q3_after + 1.5*IQR_after)).sum()
            
            # Вывод всех показателей в две колонки
            stats = [
                ('Минимум', min_before, min_after),
                ('Максимум', max_before, max_after),
                ('Среднее', mean_before, mean_after),
                ('Медиана', median_before, median_after),
                ('Стандартное отклонение', std_before, std_after),
                ('25% квантиль', q25_before, q25_after),
                ('75% квантиль', q75_before, q75_after),
            ]
            
            for name, before_val, after_val in stats:
                before_str = f"{name}: {before_val:.0f}"
                after_str = f"{name}: {after_val:.0f}"
                print(f"{before_str:<55}{after_str:<55}")
            
            # Выбросы
            before_outlier_str = f"Выбросы: {outliers_before} ({outliers_before/len(df_before)*100:.1f}%)"
            after_outlier_str = f"Выбросы: {outliers_after} ({outliers_after/len(df_after)*100:.1f}%)"
            print(f"{before_outlier_str:<55}{after_outlier_str:<55}")
    
    print("\n" + "=" * 110)
    print("Сравнительный анализ завершен")
    print("=" * 110)
    

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

    # --- Outlier Removal Function ---
def remove_outliers_iqr(df_in, col_list):
    df_out = df_in.copy()
    initial_rows = len(df_out)
    removed_indices = []
    for col in col_list:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers to be removed by this column
        outlier_indices = df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)].index
        removed_indices.extend(outlier_indices.tolist())
        
    # Remove unique outlier indices identified across all specified columns
    df_out = df_out.drop(index=list(set(removed_indices)))
    
    removed_rows = initial_rows - len(df_out)
    lg.log_event("INFO", f"Removed {removed_rows} rows (initial: {initial_rows}, remaining: {len(df_out)}) after IQR outlier treatment.")
    print(f"Initial number of rows: {initial_rows}")
    print(f"Number of rows after outlier removal: {len(df_out)}")
    print(f"Number of rows removed: {removed_rows}")
    return df_out


def compare_pairplots_subplots(df_before, df_after, spending_cols, hue='Region'):
    """
    Сравнивает pairplot ДО и ПОСЛЕ удаления выбросов (исправленная версия)
    
    Parameters:
    -----------
    df_before : DataFrame
        Данные до удаления выбросов
    df_after : DataFrame
        Данные после удаления выбросов
    spending_cols : list
        Список признаков для визуализации
    hue : str
        Признак для цветовой кодировки ('Region' или 'Channel')
    """
    
    custom_palette = ['#1E88E5', '#E53935', '#43A047']  # Голубой, Красный, Зеленый
    
    # СОЗДАЕМ ДВА ОТДЕЛЬНЫХ ГРАФИКА И РАСПОЛАГАЕМ ИХ РЯДОМ
    fig = plt.figure(figsize=(30, 12))
    
    # График ДО удаления выбросов
    print("Строим график ДО удаления выбросов...")
    g_before = sns.pairplot(df_before,
                            vars=spending_cols,
                            hue=hue,
                            palette=custom_palette,
                            diag_kind='kde',
                            height=2.5,
                            plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
    
    # Настраиваем заголовок для первого графика
    g_before.fig.suptitle('BEFORE Outlier Removal', y=1.02, fontsize=16, fontweight='bold')
    
    # График ПОСЛЕ удаления выбросов
    print("Строим график ПОСЛЕ удаления выбросов...")
    g_after = sns.pairplot(df_after,
                           vars=spending_cols,
                           hue=hue,
                           palette=custom_palette,
                           diag_kind='kde',
                           height=2.5,
                           plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5})
    
    # Настраиваем заголовок для второго графика
    g_after.fig.suptitle('AFTER Outlier Removal', y=1.02, fontsize=16, fontweight='bold')
    
    plt.show()
    
    print("\nСравнение завершено. Прокрутите вверх, чтобы увидеть оба графика.")


def boxplot(df_original, df_cleaned, cols):
    
  fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# До удаления
  sns.boxplot(data=df_original[cols], ax=axes[0])
  axes[0].set_title('ДО удаления выбросов')
  axes[0].tick_params(axis='x', rotation=45)
  axes[0].grid(True, alpha=0.3)

# После удаления
  sns.boxplot(data=df_cleaned[cols], ax=axes[1])
  axes[1].set_title('ПОСЛЕ удаления выбросов')
  axes[1].tick_params(axis='x', rotation=45)
  axes[1].grid(True, alpha=0.3)

  plt.suptitle('Сравнение диаграмм размаха до и после удаления выбросов', fontsize=14)
  plt.tight_layout()
  plt.show()

def matrix_cor(df_original, df_cleaned, cols):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Матрица корреляции ДО удаления выбросов
    correlation_matrix_original = df_original[cols].corr()
    sns.heatmap(correlation_matrix_original, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'label': 'Корреляция'}, 
                ax=axes[0], annot_kws={'size': 10})
    axes[0].set_title('Correlation Matrix of Spending Features\n(BEFORE Outlier Removal)', 
                      fontsize=12, fontweight='bold')
    
    # Матрица корреляции ПОСЛЕ удаления выбросов
    correlation_matrix_cleaned = df_cleaned[cols].corr()
    sns.heatmap(correlation_matrix_cleaned, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'label': 'Корреляция'}, 
                ax=axes[1], annot_kws={'size': 10})
    axes[1].set_title('Correlation Matrix of Spending Features\n(AFTER Outlier Removal)', 
                      fontsize=12, fontweight='bold')
    
    plt.suptitle('Сравнение корреляционных матриц', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

