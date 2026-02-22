import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logger as lg


# Настройка стилей для графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Visual_Robot:
    def __init__(self, data=None):
        self.data = data
        self.fig_size = (12, 8)
        
    def set_data(self, data):
        """Установка данных для визуализации"""
        self.data = data
        
    def plot_robots_by_industry(self, year=None):
        """
        Визуализация количества роботов по отраслям
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        if year:
            df_plot = self.data[self.data['Year'] == year]
            title = f'Количество внедренных роботов по отраслям ({year} г.)'
        else:
            df_plot = self.data.groupby('Industry')['Robots_Adopted'].sum().reset_index()
            title = 'Общее количество внедренных роботов по отраслям'
        
        # Сортировка по убыванию
        df_plot = df_plot.sort_values('Robots_Adopted', ascending=True)
        
        # Горизонтальная столбчатая диаграмма
        plt.barh(df_plot['Industry'], df_plot['Robots_Adopted'])
        plt.xlabel('Количество роботов')
        plt.ylabel('Отрасль')
        plt.title(title)
        
        # Добавление значений на график
        for i, v in enumerate(df_plot['Robots_Adopted']):
            plt.text(v + 5, i, str(v), va='center')
        
        plt.tight_layout()
        lg.log_event("INFO", f"Создан график: {title}")
        plt.show()
        
    def plot_productivity_trend(self):
        """
        Тренд производительности по годам
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # Группировка по годам
        yearly_data = self.data.groupby('Year').agg({
            'Productivity_Gain': 'mean',
            'Robots_Adopted': 'sum'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        
        # График производительности
        color = 'tab:red'
        ax1.set_xlabel('Год')
        ax1.set_ylabel('Средний рост производительности (%)', color=color)
        ax1.plot(yearly_data['Year'], yearly_data['Productivity_Gain'], 
                marker='o', color=color, linewidth=2, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Добавление значений на график
        for x, y in zip(yearly_data['Year'], yearly_data['Productivity_Gain']):
            ax1.text(x, y + 0.5, f'{y:.1f}%', ha='center', fontsize=9)
        
        # Вторая ось для количества роботов
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Общее количество роботов', color=color)
        ax2.bar(yearly_data['Year'], yearly_data['Robots_Adopted'], 
                alpha=0.3, color=color, width=0.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Динамика производительности и внедрения роботов по годам')
        plt.tight_layout()
        lg.log_event("INFO", "Создан график тренда производительности")
        plt.show()
        
    def plot_jobs_displaced_trend(self):
        """
        Тренд производительности по годам
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # Группировка по годам
        yearly_data = self.data.groupby('Year').agg({
            'Jobs_Displaced': 'mean',
            'Robots_Adopted': 'sum'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        
        # График сокащения численности
        color = 'tab:red'
        ax1.set_xlabel('Год')
        ax1.set_ylabel('Численность сокращенного персонала', color=color)
        ax1.plot(yearly_data['Year'], yearly_data['Jobs_Displaced'], 
                marker='o', color=color, linewidth=2, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Добавление значений на график
        for x, y in zip(yearly_data['Year'], yearly_data['Jobs_Displaced']):
            ax1.text(x, y + 0.5, f'{y:.1f}%', ha='center', fontsize=9)
        
        # Вторая ось для количества роботов
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Общее количество роботов', color=color)
        ax2.bar(yearly_data['Year'], yearly_data['Robots_Adopted'], 
                alpha=0.3, color=color, width=0.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Динамика сокращения численности персонала и внедрения роботов по годам')
        plt.tight_layout()
        lg.log_event("INFO", "Создан график тренда соращения персонала")
        plt.show()


    def plot_cost_savings_heatmap(self):
        """
        Тепловая карта экономии по отраслям и годам
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Создание сводной таблицы
        pivot_data = self.data.pivot_table(
            values='Cost_Savings', 
            index='Industry', 
            columns='Year', 
            aggfunc='mean',
            fill_value=0
        )
        
        # Тепловая карта
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   linewidths=0.5, cbar_kws={'label': 'Экономия (млн $)'})
        
        plt.title('Тепловая карта экономии по отраслям и годам')
        plt.xlabel('Год')
        plt.ylabel('Отрасль')
        plt.tight_layout()
        lg.log_event("INFO", "Создана тепловая карта экономии")
        plt.show()
        
    def plot_correlation_matrix(self):
        """
        Матрица корреляции между показателями
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Выбор числовых колонок
        numeric_cols = ['Robots_Adopted', 'Productivity_Gain', 
                       'Cost_Savings', 'Jobs_Displaced', 'Training_Hours']
        
        # Расчет корреляции
        corr_matrix = self.data[numeric_cols].corr()
        
        # Маска для верхнего треугольника
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Тепловая карта корреляции
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={'label': 'Корреляция'})
        
        # Переименование подписей для лучшей читаемости
        labels = ['Роботы', 'Производит.', 'Экономия', 
                 'Сокращения', 'Обучение (ч)']
        
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        
        plt.title('Матрица корреляции показателей')
        plt.tight_layout()
        lg.log_event("INFO", "Создана матрица корреляции")
        plt.show()
        
    def plot_jobs_vs_productivity(self):
        """
        Диаграмма рассеяния: сокращение персонала vs производительность
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # Цвета по отраслям
        industries = self.data['Industry'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(industries)))
        
        for industry, color in zip(industries, colors):
            mask = self.data['Industry'] == industry
            plt.scatter(self.data.loc[mask, 'Jobs_Displaced'], 
                       self.data.loc[mask, 'Productivity_Gain'],
                       label=industry, color=color, s=100, alpha=0.7)
        
        # Линия тренда
        z = np.polyfit(self.data['Jobs_Displaced'], 
                      self.data['Productivity_Gain'], 1)
        p = np.poly1d(z)
        plt.plot(self.data['Jobs_Displaced'], 
                p(self.data['Jobs_Displaced']), 
                "r--", alpha=0.5, label='Линия тренда')
        
        plt.xlabel('Количество сокращенных рабочих мест')
        plt.ylabel('Рост производительности (%)')
        plt.title('Связь между сокращением персонала и ростом производительности')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        lg.log_event("INFO", "Создана диаграмма рассеяния Jobs vs Productivity")
        plt.show()
        
    def plot_training_impact(self):
        """
        Влияние обучения на производительность
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Часы обучения по отраслям
        training_by_industry = self.data.groupby('Industry')['Training_Hours'].mean().sort_values()
        
        axes[0].barh(training_by_industry.index, training_by_industry.values)
        axes[0].set_xlabel('Среднее количество часов обучения')
        axes[0].set_title('Среднее время обучения по отраслям')
        
        # Добавление значений
        for i, v in enumerate(training_by_industry.values):
            axes[0].text(v + 5, i, f'{v:.0f} ч', va='center')
        
        # График 2: Зависимость производительности от обучения
        scatter = axes[1].scatter(self.data['Training_Hours'], 
                                  self.data['Productivity_Gain'],
                                  c=self.data['Robots_Adopted'], 
                                  cmap='viridis', s=100, alpha=0.6)
        
        axes[1].set_xlabel('Часы обучения')
        axes[1].set_ylabel('Рост производительности (%)')
        axes[1].set_title('Влияние обучения на производительность')
        plt.colorbar(scatter, ax=axes[1], label='Количество роботов')
        
        # Линия тренда
        if len(self.data) > 1:
            z = np.polyfit(self.data['Training_Hours'], 
                          self.data['Productivity_Gain'], 1)
            p = np.poly1d(z)
            axes[1].plot(self.data['Training_Hours'], 
                        p(self.data['Training_Hours']), 
                        "r--", alpha=0.5)
        
        plt.tight_layout()
        lg.log_event("INFO", "Создан график влияния обучения")
        plt.show()
        
    def plot_industry_comparison(self, metric='Productivity_Gain'):
        """
        Сравнение отраслей по выбранному показателю
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # Подготовка данных
        metrics_dict = {
            'Productivity_Gain': 'Рост производительности (%)',
            'Cost_Savings': 'Экономия (млн $)',
            'Robots_Adopted': 'Количество роботов',
            'Jobs_Displaced': 'Сокращено рабочих мест',
            'Training_Hours': 'Часы обучения'
        }
        
        if metric not in metrics_dict:
            lg.log_event("ERROR", f"Неизвестный показатель: {metric}")
            return
        
        # Группировка по отраслям
        industry_stats = self.data.groupby('Industry')[metric].agg(['mean', 'std']).reset_index()
        industry_stats = industry_stats.sort_values('mean', ascending=True)
        
        # Столбчатая диаграмма с погрешностями
        plt.barh(industry_stats['Industry'], industry_stats['mean'],
                xerr=industry_stats['std'], capsize=5, alpha=0.8)
        
        plt.xlabel(metrics_dict[metric])
        plt.title(f'Сравнение отраслей по показателю: {metrics_dict[metric]}')
        
        # Добавление значений
        for i, (mean, std) in enumerate(zip(industry_stats['mean'], industry_stats['std'])):
            plt.text(mean + std + 0.5, i, f'{mean:.1f} ± {std:.1f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        lg.log_event("INFO", f"Создан график сравнения отраслей по {metric}")
        plt.show()
        
    def plot_yearly_growth(self):
        """
        Годовой рост показателей
        """
        if self.data is None:
            lg.log_event("ERROR", "Нет данных для визуализации")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # Группировка по годам
        yearly_stats = self.data.groupby('Year').agg({
            'Robots_Adopted': 'sum',
            'Productivity_Gain': 'mean',
            'Cost_Savings': 'sum'
        }).reset_index()
        
        # Нормализация данных для сравнения
        for col in ['Robots_Adopted', 'Productivity_Gain', 'Cost_Savings']:
            yearly_stats[f'{col}_norm'] = (yearly_stats[col] / yearly_stats[col].iloc[0]) * 100
        
        # Построение графиков
        plt.plot(yearly_stats['Year'], yearly_stats['Robots_Adopted_norm'], 
                marker='o', linewidth=2, label='Роботы')
        plt.plot(yearly_stats['Year'], yearly_stats['Productivity_Gain_norm'], 
                marker='s', linewidth=2, label='Производительность')
        plt.plot(yearly_stats['Year'], yearly_stats['Cost_Savings_norm'], 
                marker='^', linewidth=2, label='Экономия')
        
        plt.xlabel('Год')
        plt.ylabel('Рост относительно базового года (%)')
        plt.title('Динамика роста показателей (базовый год = 100%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Добавление аннотаций
        for i, row in yearly_stats.iterrows():
            plt.annotate(f"{row['Year']}", 
                        (row['Year'], row['Robots_Adopted_norm']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        lg.log_event("INFO", "Создан график годового роста")
        plt.show()

# Функции для быстрого вызова (для обратной совместимости)
def create_visualizations(data):
    """
    Создание всех визуализаций
    """
    viz = RoboticsVisualizer(data)
    
    print("1. Количество роботов по отраслям")
    viz.plot_robots_by_industry()
    
    print("\n2. Тренд производительности")
    viz.plot_productivity_trend()
    
    print("\n3. Тепловая карта экономии")
    viz.plot_cost_savings_heatmap()
    
    print("\n4. Матрица корреляции")
    viz.plot_correlation_matrix()
    
    print("\n5. Связь сокращений и производительности")
    viz.plot_jobs_vs_productivity()
    
    print("\n6. Влияние обучения")
    viz.plot_training_impact()
    
    print("\n7. Сравнение отраслей")
    viz.plot_industry_comparison('Productivity_Gain')
    
    print("\n8. Годовой рост")
    viz.plot_yearly_growth()