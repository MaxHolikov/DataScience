import get_dataset as gd
import logger as lg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN

lg.clear_log_file()

# Создаем словарь для результатов
my_results_dict = {}

# ====================== 1. ИСПРАВЛЕННЫЙ КЛАСС AnomalyDetection ======================

class AnomalyDetection:
    """Класс для поиска аномалий в подготовленных данных"""
    
    def __init__(self, file_path):
        """
        Инициализация с уже очищенным датасетом
        """
        self.file_path = file_path
        self.df = None
        self.features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        self.categorical = ['Channel', 'Region']
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.anomaly_results = {}
        self.anomaly_scores = {}
        self.load_data()  # Вызываем метод загрузки
        
    def load_data(self):
        """Загрузка данных из CSV файла"""
        print("\n" + "=" * 80)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"   Файл загружен: {self.file_path}")
            print(f"   Размер: {self.df.shape}")
            print(f"   Колонки: {list(self.df.columns)}")
            
            lg.log_event("INFO", f"File loaded: {self.file_path}, shape: {self.df.shape}")
            
        except FileNotFoundError:
            print(f" Файл {self.file_path} не найден!")
            lg.log_event("ERROR", f"File not found: {self.file_path}")
            # Пробуем загрузить исходный датасет
            self.df = pd.read_csv('Wholesale customers data.csv')
            print(f" Загружен исходный датасет: {self.df.shape}")
    
    def prepare_data(self):
        """Подготовка данных для анализа аномалий"""
        print("\n" + "=" * 80)
        print("ПОДГОТОВКА ДАННЫХ")
        print("=" * 80)
        
        # Проверяем наличие признаков
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            print(f"Отсутствуют признаки: {missing_features}")
            # Используем только существующие признаки
            self.features = [f for f in self.features if f in self.df.columns]
        
        # Выделяем числовые признаки
        X = self.df[self.features].copy()
        
        # Проверка на пропуски
        if X.isnull().any().any():
            print("Обнаружены пропуски, заполняем медианой...")
            X = X.fillna(X.median())
        
        # Нормализация
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"   Данные подготовлены")
        print(f"   Размер: {self.X_scaled.shape}")
        print(f"   Признаки: {self.features}")
        print(f"   Среднее после нормализации: {np.mean(self.X_scaled, axis=0).round(2)}")
        print(f"   Стандартное отклонение: {np.std(self.X_scaled, axis=0).round(2)}")
        
        return self.X_scaled


# ====================== 2. КЛАСС AnomalyDetector (из вашего кода) ======================

class AnomalyDetector:
    """Класс для обнаружения аномалий различными методами"""
    
    def __init__(self, data_loader):
        self.data = data_loader
        self.results = {}
        self.scores = {}
        self.consensus = None
        
    def detect_isolation_forest(self, contamination=0.1):
        """Метод 1: Isolation Forest"""
        print("\n Isolation Forest...")
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = model.fit_predict(self.data.X_scaled)
        scores = model.score_samples(self.data.X_scaled)
        
        anomaly_mask = predictions == -1
        self.results['Isolation Forest'] = anomaly_mask
        self.scores['Isolation Forest'] = scores
        
        n_anomalies = np.sum(anomaly_mask)
        print(f"    Найдено аномалий: {n_anomalies} ({n_anomalies/len(self.data.X_scaled)*100:.2f}%)")
        
        return anomaly_mask
    
    def detect_lof(self, contamination=0.1):
        """Метод 2: Local Outlier Factor"""
        print("\n Local Outlier Factor...")
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
        predictions = model.fit_predict(self.data.X_scaled)
        
        anomaly_mask = predictions == -1
        self.results['LOF'] = anomaly_mask
        
        n_anomalies = np.sum(anomaly_mask)
        print(f"    Найдено аномалий: {n_anomalies} ({n_anomalies/len(self.data.X_scaled)*100:.2f}%)")
        
        return anomaly_mask
    
    def detect_one_class_svm(self, contamination=0.1):
        """Метод 3: One-Class SVM"""
        print("\n One-Class SVM...")
        model = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
        predictions = model.fit_predict(self.data.X_scaled)
        
        anomaly_mask = predictions == -1
        self.results['One-Class SVM'] = anomaly_mask
        
        n_anomalies = np.sum(anomaly_mask)
        print(f"    Найдено аномалий: {n_anomalies} ({n_anomalies/len(self.data.X_scaled)*100:.2f}%)")
        
        return anomaly_mask
    
    def detect_elliptic_envelope(self, contamination=0.1):
        """Метод 4: Elliptic Envelope"""
        print("\n Elliptic Envelope...")
        try:
            model = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = model.fit_predict(self.data.X_scaled)
            
            anomaly_mask = predictions == -1
            self.results['Elliptic Envelope'] = anomaly_mask
            
            n_anomalies = np.sum(anomaly_mask)
            print(f"    Найдено аномалий: {n_anomalies} ({n_anomalies/len(self.data.X_scaled)*100:.2f}%)")
        except Exception as e:
            print(f"    Elliptic Envelope не применим: {e}")
            self.results['Elliptic Envelope'] = np.zeros(len(self.data.X_scaled), dtype=bool)
        
        return self.results['Elliptic Envelope']
    
    def detect_dbscan(self, eps=0.5, min_samples=5):
        """Метод 5: DBSCAN (шум = аномалии)"""
        print("\n DBSCAN...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(self.data.X_scaled)
        
        anomaly_mask = clusters == -1
        self.results['DBSCAN'] = anomaly_mask
        
        n_anomalies = np.sum(anomaly_mask)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        print(f"   Найдено кластеров: {n_clusters}")
        print(f"   Найдено аномалий (шум): {n_anomalies} ({n_anomalies/len(self.data.X_scaled)*100:.2f}%)")
        
        return anomaly_mask
    
    def detect_all_methods(self, contamination=0.1):
        """Запуск всех методов обнаружения"""
        print("\n" + "=" * 80)
        print("ПОИСК АНОМАЛИЙ (5+ МЕТОДОВ)")
        print("=" * 80)
        print(f"Размер данных: {len(self.data.X_scaled)} записей")
        print(f"Параметр contamination: {contamination}")
        
        self.detect_isolation_forest(contamination)
        self.detect_lof(contamination)
        self.detect_one_class_svm(contamination)
        self.detect_elliptic_envelope(contamination)
        self.detect_dbscan(eps=0.5, min_samples=5)
        
        # Вычисляем согласованность
        self._calculate_consensus()
        
        return self.results
    
    def _calculate_consensus(self):
        """Вычисление согласованности методов"""
        if not self.results:
            return
        
        consensus = np.zeros(len(self.data.X_scaled))
        for name, mask in self.results.items():
            consensus += mask.astype(int)
        
        self.consensus = consensus
        self.consensus_anomalies = consensus >= 3  # Аномалия, если обнаружена 3+ методами
        
        print("\n" + "-" * 60)
        print("   СОГЛАСОВАННОСТЬ МЕТОДОВ:")
        print(f"   Всего записей: {len(self.data.X_scaled)}")
        print(f"   Аномалии (3+ метода): {np.sum(self.consensus_anomalies)}")
        print(f"   Аномалии (2 метода): {np.sum(consensus == 2)}")
        print(f"   Аномалии (1 метод): {np.sum(consensus == 1)}")
        
        # Сохраняем результаты в датафрейм
        self.data.df['Anomaly_Score'] = consensus
        self.data.df['Is_Anomaly'] = self.consensus_anomalies


# ====================== 3. КЛАСС AnomalyVisualizer (сокращенная версия) ======================

class AnomalyVisualizer:
    """Класс для визуализации результатов поиска аномалий"""
    
    def __init__(self, detector):
        self.detector = detector
        self.data = detector.data
        
    def plot_all_graphs(self):
        """Построение всех графиков"""
        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ ВСЕХ ГРАФИКОВ")
        print("=" * 80)
        
        self.plot_pca_anomalies()
        self.plot_anomaly_distribution()
        self.plot_anomaly_scores()
        self.plot_anomaly_details()
        self.plot_feature_correlation()
        self.plot_anomaly_profile()
        
    def plot_pca_anomalies(self):
        """График 1: PCA визуализация аномалий"""
        print("\n График 1: PCA визуализация аномалий")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.data.X_scaled)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, mask) in enumerate(self.detector.results.items()):
            if idx >= len(axes) - 1:
                break
            ax = axes[idx]
            
            ax.scatter(X_pca[~mask, 0], X_pca[~mask, 1], 
                      c='blue', alpha=0.5, s=30, label='Normal', 
                      edgecolor='black', linewidth=0.3)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c='red', alpha=0.9, s=100, marker='X', 
                      label=f'Anomalies ({np.sum(mask)})', 
                      edgecolor='black', linewidth=1)
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        ax = axes[5]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                       c=self.detector.consensus, cmap='RdYlGn_r', 
                       s=50, alpha=0.7, edgecolor='black', linewidth=0.3,
                       vmin=0, vmax=len(self.detector.results))
        plt.colorbar(sc, ax=ax, label='Number of methods detecting anomaly')
        ax.set_title('Consensus (3+ methods = anomaly)', fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Anomaly Detection Results (5 methods)', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
  
    
    def plot_anomaly_distribution(self):
        """График 2: Распределение аномалий по признакам"""
        print("\n График 2: Распределение аномалий по признакам...")
        
        if 'Is_Anomaly' not in self.data.df.columns:
            return
        
        normal_mean = self.data.df[~self.data.df['Is_Anomaly']][self.data.features].mean()
        anomaly_mean = self.data.df[self.data.df['Is_Anomaly']][self.data.features].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        x = range(len(self.data.features))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], normal_mean.values, width, 
                   label='Normal', color='blue', alpha=0.7, edgecolor='black')
        axes[0].bar([i + width/2 for i in x], anomaly_mean.values, width, 
                   label='Anomaly', color='red', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Normal vs Anomaly: Mean Values', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.data.features, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        if 'Channel' in self.data.df.columns:
            channel_anomalies = pd.crosstab(
                self.data.df['Channel'].map({1: 'Horeca', 2: 'Retail'}),
                self.data.df['Is_Anomaly'].map({True: 'Anomaly', False: 'Normal'})
            )
            channel_anomalies.plot(kind='bar', ax=axes[1], 
                                  color=['blue', 'red'], edgecolor='black')
            axes[1].set_title('Anomalies by Channel', fontweight='bold')
            axes[1].set_xlabel('Channel')
            axes[1].set_ylabel('Count')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_scores(self):
        """График 3: Распределение аномальных оценок"""
        print("\n График 3: Распределение аномальных оценок...")
        
        if 'Is_Anomaly' not in self.data.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.detector.consensus, bins=range(len(self.detector.results) + 2), 
                    edgecolor='black', color='skyblue', alpha=0.7, align='left')
        axes[0].set_xlabel('Number of methods detecting anomaly')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Anomaly Scores', fontweight='bold')
        axes[0].set_xticks(range(len(self.detector.results) + 1))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        anomaly_count = self.data.df['Is_Anomaly'].sum()
        normal_count = len(self.data.df) - anomaly_count
        
        axes[1].pie([normal_count, anomaly_count], 
                   labels=['Normal', 'Anomaly'],
                   autopct='%1.1f%%',
                   colors=['blue', 'red'],
                   explode=(0, 0.1),
                   startangle=90)
        axes[1].set_title('Final Anomaly Detection Results', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_details(self, n_samples=5):
        """График 4: Детальный показ аномальных записей"""
        print("\n График 4: Детальный показ аномальных записей...")
        
        if 'Is_Anomaly' not in self.data.df.columns:
            return
        
        anomalies = self.data.df[self.data.df['Is_Anomaly']]
        
        if len(anomalies) == 0:
            print(" Аномалии не найдены")
            return
        
        print(f"Всего аномалий: {len(anomalies)}")
        print(f"\nТоп-{min(n_samples, len(anomalies))} аномальных записей:")
        print(anomalies.head(n_samples).to_string())
        
        # Визуализация топ аномалий
        fig, ax = plt.subplots(figsize=(14, 6))
        
        top_anomalies = anomalies.head(10)
        x = range(len(top_anomalies))
        width = 0.1
        
        for i, feature in enumerate(self.data.features):
            ax.bar([xi + i*width for xi in x], top_anomalies[feature].values, 
                   width, label=feature, alpha=0.8)
        
        ax.set_xlabel('Anomaly Index')
        ax.set_ylabel('Value')
        ax.set_title('Top 10 Anomalies - Feature Values', fontweight='bold')
        ax.set_xticks([xi + 2.5*width for xi in x])
        ax.set_xticklabels(top_anomalies.index)
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlation(self):
        """График 5: Корреляция признаков с аномалиями"""
        print("\n График 5: Корреляция признаков с аномалиями...")
        
        if 'Is_Anomaly' not in self.data.df.columns:
            return
        
        correlations = []
        for feature in self.data.features:
            corr = self.data.df[feature].corr(self.data.df['Is_Anomaly'].astype(int))
            correlations.append({'feature': feature, 'correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x > 0 else 'blue' for x in corr_df['correlation']]
        plt.barh(corr_df['feature'], corr_df['correlation'], 
                color=colors, edgecolor='black')
        plt.xlabel('Correlation with Anomaly')
        plt.title('Feature Correlation with Anomaly Detection', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_profile(self):
        """График 6: Профиль аномальных записей"""
        print("\n График 6: Профиль аномальных записей...")
        
        if 'Is_Anomaly' not in self.data.df.columns:
            return
        
        X_norm = (self.data.df[self.data.features] - self.data.df[self.data.features].mean()) / self.data.df[self.data.features].std()
        
        normal_profile = X_norm[~self.data.df['Is_Anomaly']].mean()
        anomaly_profile = X_norm[self.data.df['Is_Anomaly']].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(self.data.features))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], normal_profile.values, width, 
               label='Normal', color='blue', alpha=0.7, edgecolor='black')
        ax.bar([i + width/2 for i in x], anomaly_profile.values, width, 
               label='Anomaly', color='red', alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Standard Deviation from Mean')
        ax.set_title('Anomaly Profile: Deviation from Normal', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.data.features, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()