import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                           adjusted_rand_score, normalized_mutual_info_score, completeness_score,
                           homogeneity_score, v_measure_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import logger as lg

class WholesaleClustering:

    '''
    Класс для кластеризации датасета Wholesale Customers
        Атрибуты:
    -----------
    df : pandas.DataFrame - Исходный датасет
    features : list
    Список признаков для кластеризации
    X_scaled : numpy.array
    Нормализованные данные
    results : dict
        Результаты кластеризации для каждого алгоритма
    metrics : dict
    '''
  
    def __init__(self, data=None, file_path='Wholesale customers data.csv'):
      self.file_path = file_path
      self.df = data  # Теперь можно передать готовый DataFrame
      self.X = None
      self.X_scaled = None
      self.features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
      self.scaler = None
      self.results = {}
      self.all_labels = {}
      self.metrics = {
            'ari_scores': {},
            'nmi_scores': {},
            'completeness_scores': {},
            'homogeneity_scores': {},
            'v_measure_scores': {}
        }
      self.metrics_df = None
      self.n_clusters = 4
      self.my_results_dict = {}

        # Если данные не переданы, загружаем из файла
      if self.df is None:
        self.load_data()
      
    def prepare_data(self):
      
        """Подготавливает данные для кластеризации"""
        # Признаки для кластеризации
        self.features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        
        # Проверяем наличие признаков
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")
        
      # Выбираем признаки
        self.X = self.df[self.features].copy()
      # Нормализация
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Вывод информации
        log_message=(
                    f"Количество записей: {self.X.shape[0]}\n" +
                    f"Количество признаков: {self.X.shape[1]}\n"+
                    f"Признаки: {self.features}\n"+
                    f"Размер после нормализации: {self.X_scaled.shape}\n"+
                    f"Минимальное значение: {np.min(self.X_scaled):.2f}\n"+
                    f"Максимальное значение: {np.max(self.X_scaled):.2f}")
        lg.log_event("INFO", log_message)            
        # Сохраняем в словарь
        self.my_results_dict['X'] = self.X
        self.my_results_dict['X_scaled'] = self.X_scaled
        self.my_results_dict['features'] = self.features
        self.my_results_dict['scaler'] = self.scaler
        
        #return self.X_scaled
  

    def find_optimal_clusters(self, max_clusters=10, show_heatmaps=True):
      """
      Определяет оптимальное число кластеров с визуализацией метрик и тепловых карт
    
      Parameters:
      -----------
      max_clusters : int
        Максимальное число кластеров для оценки
      show_heatmaps : bool
        Показывать ли тепловые карты для каждого k
    
      Returns:
      --------
      int : Оптимальное число кластеров
      """
      inertias = []
      sil_scores = []
      calinski_scores = []
      davies_scores = []
    
      K_range = range(2, max_clusters+1)
    
      # Если нужно показать тепловые карты, создаем отдельную фигуру
      if show_heatmaps:
        heatmap_fig, heatmap_axes = plt.subplots(2, 3, figsize=(18, 12))
        heatmap_axes = heatmap_axes.flatten()
        heatmap_idx = 0
    
      for k in K_range:
        print(f"\nОценка для k = {k}")
        
        # Кластеризация
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_scaled)
        
        # Сохраняем метрики
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(self.X_scaled, labels)
        sil_scores.append(sil)
        calinski_scores.append(calinski_harabasz_score(self.X_scaled, labels))
        davies_scores.append(davies_bouldin_score(self.X_scaled, labels))
        
        print(f"   Silhouette: {sil:.4f}")
        
        # ТЕПЛОВАЯ КАРТА для текущего k
        if show_heatmaps and heatmap_idx < len(heatmap_axes):
          ax = heatmap_axes[heatmap_idx]
            
          # Сортируем образцы по меткам кластеров
          sorted_indices = np.argsort(labels)
          X_sorted = self.X_scaled[sorted_indices]
            
          # Рисуем тепловую карту
          sns.heatmap(X_sorted, cmap='viridis', cbar=True, 
                       ax=ax, xticklabels=self.features, yticklabels=False)
            
          # Добавляем разделители между кластерами
          cluster_boundaries = []
          current_label = labels[sorted_indices[0]]
          for i, label in enumerate(labels[sorted_indices]):
            if label != current_label:
              cluster_boundaries.append(i)
              current_label = label
            
          for boundary in cluster_boundaries:
            ax.axhline(y=boundary, color='red', linewidth=2, linestyle='--')
            
          ax.set_title(f'k={k} (Silhouette={sil:.3f})')
          ax.set_xlabel('Признаки')
          ax.set_ylabel('Образцы (отсортированы по кластерам)')
            
          heatmap_idx += 1
    
    # Скрываем неиспользованные оси тепловых карт
      if show_heatmaps:
        for idx in range(heatmap_idx, len(heatmap_axes)):
          heatmap_axes[idx].set_visible(False)
        
        heatmap_fig.suptitle('Тепловые карты кластеров для разного числа кластеров', 
                            y=1.02, fontsize=14)
        heatmap_fig.tight_layout()
        plt.show()
    
    # Визуализация метрик
      fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Метод локтя
      axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
      axes[0, 0].set_xlabel('Количество кластеров')
      axes[0, 0].set_ylabel('Inertia')
      axes[0, 0].set_title('Метод локтя')
      axes[0, 0].grid(True, alpha=0.3)
    
    # Оптимальное по локтю (находим точку перегиба)
      if len(inertias) > 2:
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        elbow_point = np.argmax(diff_diffs) + 2
        axes[0, 0].axvline(x=elbow_point+2, color='green', linestyle='--', 
                           label=f'Локоть: k={elbow_point+2}')
        axes[0, 0].legend()
    
    # Silhouette
      axes[0, 1].plot(K_range, sil_scores, 'ro-', linewidth=2, markersize=8)
      axes[0, 1].set_xlabel('Количество кластеров')
      axes[0, 1].set_ylabel('Silhouette Score')
      axes[0, 1].set_title('Коэффициент силуэта')
      axes[0, 1].grid(True, alpha=0.3)
      max_sil_idx = np.argmax(sil_scores)
      axes[0, 1].axvline(x=max_sil_idx+2, color='red', linestyle='--', 
                       label=f'Максимум: k={max_sil_idx+2}')
      axes[0, 1].legend()
    
    # Calinski-Harabasz
      axes[1, 0].plot(K_range, calinski_scores, 'go-', linewidth=2, markersize=8)
      axes[1, 0].set_xlabel('Количество кластеров')
      axes[1, 0].set_ylabel('Calinski-Harabasz Score')
      axes[1, 0].set_title('Индекс Калински-Харабаса')
      axes[1, 0].grid(True, alpha=0.3)
      max_cal_idx = np.argmax(calinski_scores)
      axes[1, 0].axvline(x=max_cal_idx+2, color='green', linestyle='--', 
                       label=f'Максимум: k={max_cal_idx+2}')
      axes[1, 0].legend()
    
    # Davies-Bouldin
      axes[1, 1].plot(K_range, davies_scores, 'mo-', linewidth=2, markersize=8)
      axes[1, 1].set_xlabel('Количество кластеров')
      axes[1, 1].set_ylabel('Davies-Bouldin Score')
      axes[1, 1].set_title('Индекс Дэвиса-Болдина (меньше = лучше)')
      axes[1, 1].grid(True, alpha=0.3)
      min_db_idx = np.argmin(davies_scores)
      axes[1, 1].axvline(x=min_db_idx+2, color='purple', linestyle='--', 
                       label=f'Минимум: k={min_db_idx+2}')
      axes[1, 1].legend()
    
      plt.tight_layout()
      plt.suptitle('Определение оптимального числа кластеров', y=1.02, fontsize=14)
      plt.show()
    
    # Определяем оптимальное число по большинству метрик
      votes = {}
      for k in K_range:
        votes[k] = 0
    
      # Silhouette (чем выше, тем лучше)
      best_sil_k = K_range[np.argmax(sil_scores)]
      votes[best_sil_k] += 1
    
    # Calinski-Harabasz (чем выше, тем лучше)
      best_cal_k = K_range[np.argmax(calinski_scores)]
      votes[best_cal_k] += 1

    # Davies-Bouldin (чем ниже, тем лучше)
      best_db_k = K_range[np.argmin(davies_scores)]
      votes[best_db_k] += 1
    
    # Метод локтя (точка перегиба)
      if len(inertias) > 2:
        elbow_k = np.argmax(np.diff(np.diff(inertias))) + 2
        votes[elbow_k] += 0.5  # полголоса
    
    # Находим оптимальное по голосованию
      self.n_clusters = max(votes, key=votes.get)

      print(f"\n Оптимальное число кластеров: {self.n_clusters}")
    
    # Показываем итоговую тепловую карту для оптимального k
      print(f"\n Детальный анализ для оптимального k={self.n_clusters}")
    
    # Кластеризация с оптимальным k
      final_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
      final_labels = final_kmeans.fit_predict(self.X_scaled)
    
    # Создаем улучшенную тепловую карту для оптимального k
      fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    
    # Тепловая карта
      sorted_indices = np.argsort(final_labels)
      X_sorted = self.X_scaled[sorted_indices]
      sorted_labels = final_labels[sorted_indices]
    
      sns.heatmap(X_sorted, cmap='viridis', cbar=True, 
               ax=axes[0], xticklabels=self.features, yticklabels=False)
    
    # Добавляем разделители между кластерами
      cluster_boundaries = []
      current_label = sorted_labels[0]
      for i, label in enumerate(sorted_labels):
        if label != current_label:
            cluster_boundaries.append(i)
            current_label = label
    
      for boundary in cluster_boundaries:
        axes[0].axhline(y=boundary, color='red', linewidth=2, linestyle='--')
    
      axes[0].set_title(f'Тепловая карта (k={self.n_clusters})')
      axes[0].set_xlabel('Признаки')
      axes[0].set_ylabel('Образцы (отсортированы по кластерам)')
    
    # Добавляем аннотации с размерами кластеров
      y_pos = 0
      for i, boundary in enumerate(cluster_boundaries + [len(final_labels)]):
        cluster_size = boundary - y_pos
        axes[0].text(len(self.features) + 1, (y_pos + boundary)/2, 
                    f'Кластер {i}',  #f'Кластер {i}\n(n={cluster_size})', 
                    verticalalignment='center', fontsize=6)
        y_pos = boundary
    
    # Профили кластеров
      cluster_profiles = pd.DataFrame(final_kmeans.cluster_centers_, columns=self.features)
      sns.heatmap(cluster_profiles.T, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=axes[1], cbar=True)
      axes[1].set_title(f'Профили кластеров (центры)')
      axes[1].set_xlabel('Кластер')
      axes[1].set_ylabel('Признаки')
    
      plt.tight_layout()
      plt.show()
    
      self.my_results_dict['optimal_clusters'] = self.n_clusters
      self.my_results_dict['cluster_votes'] = votes
      self.my_results_dict['final_kmeans'] = final_kmeans
      self.my_results_dict['final_labels'] = final_labels
    
      return self.n_clusters

class SegmentationAnalysis:
    """Класс для сегментации клиентов"""
    
    def __init__(self, file_path='wholesale_customers_cleaned.csv'):
        self.file_path = file_path
        self.df = None
        self.features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.clustering_results = {}
        self.metrics = {}
        self.optimal_k = None
        self.load_data()
        
    def load_data(self):
        """Загрузка подготовленного датасета"""
        print("\n" + "=" * 80)
        print(" ЗАГРУЗКА ПОДГОТОВЛЕННОГО ДАТАСЕТА")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"   Данные загружены: {self.file_path}")
            print(f"   Размер: {self.df.shape}")
            print(f"   Колонки: {list(self.df.columns)}")
            print(f"   Пропуски: {self.df.isnull().sum().sum()}")
            
            lg.log_event("INFO", f"Data loaded: {self.df.shape}")
        except FileNotFoundError:
            print(f"Файл не найден, загружаем исходный датасет")
            self.df = pd.read_csv('Wholesale customers data.csv')
            print(f"Загружен исходный датасет: {self.df.shape}")
    
    def prepare_data(self):
        """Подготовка данных для кластеризации"""
        print("\n" + "=" * 80)
        print("ПОДГОТОВКА ДАННЫХ")
        print("=" * 80)
        
        # Проверка наличия признаков
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            print(f"Отсутствуют признаки: {missing_features}")
            self.features = [f for f in self.features if f in self.df.columns]
        
        X = self.df[self.features].copy()
        
        # Нормализация
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"   Данные подготовлены")
        print(f"   Размер: {self.X_scaled.shape}")
        print(f"   Признаки: {self.features}")
        print(f"   Среднее: {np.mean(self.X_scaled, axis=0).round(2)}")
        print(f"   Стд: {np.std(self.X_scaled, axis=0).round(2)}")
        
        return self.X_scaled


# ====================== 2. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ ======================

class OptimalKFinder:
    """Класс для определения оптимального числа кластеров"""
    
    def __init__(self, segmentation):
        self.seg = segmentation
        self.K_range = range(2, 11)
        self.inertias = []
        self.silhouettes = []
        self.calinski_scores = []
        self.davies_scores = []
        
    def calculate_metrics(self):
        """Вычисление метрик для разного числа кластеров"""
        print("\n" + "=" * 80)
        print("ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ")
        print("=" * 80)
        
        for k in self.K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.seg.X_scaled)
            
            self.inertias.append(kmeans.inertia_)
            self.silhouettes.append(silhouette_score(self.seg.X_scaled, labels))
            self.calinski_scores.append(calinski_harabasz_score(self.seg.X_scaled, labels))
            self.davies_scores.append(davies_bouldin_score(self.seg.X_scaled, labels))
            
            print(f"   k={k}: Silhouette={self.silhouettes[-1]:.4f}")
        
        # Оптимальное число по большинству метрик
        self.seg.optimal_k = self.K_range[np.argmax(self.silhouettes)]
        print(f"\n Оптимальное число кластеров: {self.seg.optimal_k}")
        
        return self.seg.optimal_k
    
    def plot_metrics(self):
        """Визуализация метрик для определения оптимального k"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Метод локтя
        axes[0, 0].plot(self.K_range, self.inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette
        axes[0, 1].plot(self.K_range, self.silhouettes, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=self.seg.optimal_k, color='red', linestyle='--', alpha=0.5)
        
        # Calinski-Harabasz
        axes[1, 0].plot(self.K_range, self.calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Index', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[1, 1].plot(self.K_range, self.davies_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Index (lower is better)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Optimal Number of Clusters Determination', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()


# ====================== 3. 6 АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ ======================

class ClusteringAlgorithms:
    """Класс для применения различных алгоритмов кластеризации"""
    
    def __init__(self, segmentation):
        self.seg = segmentation
        self.algorithms = {
            'K-Means': KMeans(n_clusters=self.seg.optimal_k, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=self.seg.optimal_k),
            'Gaussian Mixture': GaussianMixture(n_components=self.seg.optimal_k, random_state=42),
            'Spectral': SpectralClustering(n_clusters=self.seg.optimal_k, random_state=42),
            'Birch': Birch(n_clusters=self.seg.optimal_k),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        self.results = {}
        self.metrics = {}
        
    def apply_all(self):
        """Применение всех алгоритмов"""
        print("\n" + "=" * 80)
        print("ПРИМЕНЕНИЕ 6 АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ")
        print("=" * 80)
        
        for name, algorithm in self.algorithms.items():
            print(f"\n {name}...")
            
            try:
                if name == 'Gaussian Mixture':
                    labels = algorithm.fit_predict(self.seg.X_scaled)
                elif name == 'DBSCAN':
                    labels = algorithm.fit_predict(self.seg.X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = sum(labels == -1)
                    print(f"   Кластеров: {n_clusters}, Шум: {n_noise}")
                else:
                    labels = algorithm.fit_predict(self.seg.X_scaled)
                
                self.results[name] = labels
                
                # Вычисление метрик
                if len(set(labels)) > 1 and not (len(set(labels)) == 2 and -1 in labels):
                    sil = silhouette_score(self.seg.X_scaled, labels)
                    cal = calinski_harabasz_score(self.seg.X_scaled, labels)
                    dav = davies_bouldin_score(self.seg.X_scaled, labels)
                    
                    self.metrics[name] = {
                        'silhouette': sil,
                        'calinski_harabasz': cal,
                        'davies_bouldin': dav,
                        'n_clusters': len(set(labels))
                    }
                    
                    print(f"    Silhouette: {sil:.4f}")
                    print(f"    Calinski-Harabasz: {cal:.2f}")
                    print(f"    Davies-Bouldin: {dav:.4f}")
                else:
                    print(f"    Не удалось вычислить метрики")
                    
            except Exception as e:
                print(f"    Ошибка: {e}")
        
        return self.results, self.metrics


# ====================== 4. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ ======================

class ClusteringVisualizer:
    """Класс для визуализации результатов кластеризации"""
    
    def __init__(self, segmentation, results, metrics):
        self.seg = segmentation
        self.results = results
        self.metrics = metrics
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(segmentation.X_scaled)
        
    def plot_clusters_pca(self):
        """PCA визуализация кластеров для каждого алгоритма"""
        n_algorithms = len(self.results)
        n_cols = 3
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_algorithms > 1 else [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.seg.optimal_k))
        
        for idx, (name, labels) in enumerate(self.results.items()):
            ax = axes[idx]
            
            for cluster in np.unique(labels):
                if cluster == -1:
                    mask = labels == cluster
                    ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                              c='black', marker='x', s=50, label='Noise', alpha=0.5)
                else:
                    mask = labels == cluster
                    ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                              c=[colors[cluster % len(colors)]], 
                              label=f'Cluster {cluster}', alpha=0.7, s=30,
                              edgecolor='black', linewidth=0.3)
            
            ax.set_title(f'{name}\n(n_clusters={len(set(labels))})', fontweight='bold')
            ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%})')
            ax.grid(True, alpha=0.3)
        
        # Скрываем пустые графики
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Clustering Algorithms Comparison', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Сравнение метрик алгоритмов"""
        if not self.metrics:
            print("Нет метрик для визуализации")
            return
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.sort_values('silhouette', ascending=False)
        
        print("\n Сравнение метрик:")
        print(metrics_df.round(4).to_string())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Silhouette
        axes[0].barh(metrics_df.index, metrics_df['silhouette'], 
                    color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Silhouette Score')
        axes[0].set_title('Silhouette Score (higher is better)', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Calinski-Harabasz
        axes[1].barh(metrics_df.index, metrics_df['calinski_harabasz'], 
                    color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Calinski-Harabasz Index')
        axes[1].set_title('Calinski-Harabasz Index (higher is better)', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Davies-Bouldin
        axes[2].barh(metrics_df.index, metrics_df['davies_bouldin'], 
                    color='salmon', edgecolor='black')
        axes[2].set_xlabel('Davies-Bouldin Index')
        axes[2].set_title('Davies-Bouldin Index (lower is better)', fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Clustering Metrics Comparison', y=1.05, fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return metrics_df
    
    def plot_silhouette_analysis(self):
        """Silhouette анализ для лучшего алгоритма"""
        if not self.metrics:
            return
        
        best_algo = max(self.metrics, key=lambda x: self.metrics[x]['silhouette'])
        labels = self.results[best_algo]
        
        from sklearn.metrics import silhouette_samples
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        silhouette_vals = silhouette_samples(self.seg.X_scaled, labels)
        y_lower = 10
        
        for i in range(self.seg.optimal_k):
            ith_cluster_silhouette = silhouette_vals[labels == i]
            ith_cluster_silhouette.sort()
            
            size_cluster = len(ith_cluster_silhouette)
            y_upper = y_lower + size_cluster
            
            color = plt.cm.tab10(i / self.seg.optimal_k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette,
                            facecolor=color, edgecolor='black', alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
            y_lower = y_upper + 10
        
        ax.axvline(x=self.metrics[best_algo]['silhouette'], color="red", linestyle="--")
        ax.set_title(f'Silhouette Plot for {best_algo}', fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_profiles(self):
        """Профили кластеров для лучшего алгоритма"""
        if not self.metrics:
            return
        
        best_algo = max(self.metrics, key=lambda x: self.metrics[x]['silhouette'])
        labels = self.results[best_algo]
        
        df_with_clusters = self.seg.df.copy()
        df_with_clusters['Cluster'] = labels
        
        cluster_profiles = df_with_clusters.groupby('Cluster')[self.seg.features].mean()
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Абсолютные значения
        sns.heatmap(cluster_profiles.T, annot=True, fmt='.0f', 
                   cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Mean Value'})
        axes[0].set_title(f'{best_algo}: Absolute Cluster Profiles', fontweight='bold')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Feature')
        
        # Нормализованные профили
        sns.heatmap(cluster_profiles_norm.T, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=axes[1], 
                   cbar_kws={'label': 'Std Deviation'})
        axes[1].set_title(f'{best_algo}: Normalized Cluster Profiles', fontweight='bold')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Feature')
        
        plt.tight_layout()
        plt.show()
        
        return cluster_profiles


# ====================== 5. МОДЕЛЬ ДЛЯ ОЦЕНКИ ВАЖНОСТИ ПРИЗНАКОВ ======================

class FeatureImportanceAnalyzer:
    """Класс для анализа важности признаков"""
    
    def __init__(self, segmentation, results):
        self.seg = segmentation
        self.results = results
        self.model = None
        self.feature_importance = None
        
    def analyze_importance(self):
        """Анализ важности признаков с помощью Random Forest"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("=" * 80)
        
        if not self.results:
            print(" Нет результатов кластеризации")
            return
        
        # Используем лучший алгоритм
        metrics = {}
        for name, labels in self.results.items():
            if len(set(labels)) > 1:
                metrics[name] = silhouette_score(self.seg.X_scaled, labels)
        
        if not metrics:
            return
        
        best_algo = max(metrics, key=metrics.get)
        y = self.results[best_algo]
        
        print(f"\nИспользуем кластеры от {best_algo} как целевую переменную")
        print(f"Распределение: {np.bincount(y[y >= 0])}")
        
        # Обучение Random Forest
        X_train, X_test, y_train, y_test = train_test_split(
            self.seg.X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n Точность модели: {accuracy:.4f}")
        print("\n Отчет по классификации:")
        print(classification_report(y_test, y_pred))
        
        # Важность признаков
        self.feature_importance = pd.DataFrame({
            'feature': self.seg.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, self.seg.X_scaled, y, cv=5)
        print(f"\n Кросс-валидация: {cv_scores}")
        print(f"   Среднее: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.feature_importance
    
    def plot_feature_importance(self):
        """Визуализация важности признаков"""
        if self.feature_importance is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Горизонтальная бар-диаграмма
        axes[0].barh(self.feature_importance['feature'], self.feature_importance['importance'],
                    color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Feature Importance (Random Forest)', fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Круговая диаграмма
        axes[1].pie(self.feature_importance['importance'], 
                   labels=self.feature_importance['feature'],
                   autopct='%1.1f%%',
                   startangle=140,
                   colors=plt.cm.viridis(np.linspace(0, 1, len(self.feature_importance))))
        axes[1].set_title('Feature Importance Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n Важность признаков:")
        print(self.feature_importance.to_string(index=False))
    
    def plot_confusion_matrix(self):
        """Матрица ошибок для лучшей модели"""
        if self.model is None:
            return
        
        # Предсказания на всех данных
        y_pred = self.model.predict(self.seg.X_scaled)
        y_true = self.results[max(self.results, 
                                  key=lambda x: silhouette_score(self.seg.X_scaled, self.results[x]) 
                                  if len(set(self.results[x])) > 1 else 0)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(set(y_true)),
                   yticklabels=sorted(set(y_true)))
        plt.title('Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()


# ====================== 6. ЗАПУСК ВСЕГО АНАЛИЗА ======================

def run_segmentation_analysis():
    """Запуск полного анализа сегментации"""
    print(" АНАЛИЗ СЕГМЕНТАЦИИ")
    print("=" * 80)
    # 1. Загрузка и подготовка данных
    seg = SegmentationAnalysis('wholesale_customers_cleaned.csv')
    seg.prepare_data()
    
    # 2. Определение оптимального числа кластеров
    k_finder = OptimalKFinder(seg)
    k_finder.calculate_metrics()
    k_finder.plot_metrics()
    
    # 3. Применение 6 алгоритмов
    clusterer = ClusteringAlgorithms(seg)
    results, metrics = clusterer.apply_all()
    
    # 4. Визуализация результатов
    visualizer = ClusteringVisualizer(seg, results, metrics)
    visualizer.plot_clusters_pca()
    metrics_df = visualizer.plot_metrics_comparison()
    visualizer.plot_silhouette_analysis()
    profiles = visualizer.plot_cluster_profiles()
    
    # 5. Анализ важности признаков
    importance_analyzer = FeatureImportanceAnalyzer(seg, results)
    importance = importance_analyzer.analyze_importance()
    importance_analyzer.plot_feature_importance()
    importance_analyzer.plot_confusion_matrix()
    
    # 6. Сохранение результатов
    best_algo = max(metrics, key=lambda x: metrics[x]['silhouette'])
    df_with_clusters = seg.df.copy()
    df_with_clusters['Cluster'] = results[best_algo]
    df_with_clusters.to_csv('wholesale_segmented.csv', index=False)
    print(f"\n Результаты сохранены в 'wholesale_segmented.csv'")
    
    # 7. Итоговый отчет
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ ПО СЕГМЕНТАЦИИ")
    print("=" * 80)
    
    print(f"""
    Всего записей: {len(seg.df)}
    Признаки: {seg.features}
    Оптимальное число кластеров: {seg.optimal_k}
    
    ЛУЧШИЙ АЛГОРИТМ: {best_algo}
    
    МЕТРИКИ:
    Silhouette Score: {metrics[best_algo]['silhouette']:.4f}
    Calinski-Harabasz: {metrics[best_algo]['calinski_harabasz']:.2f}
    Davies-Bouldin: {metrics[best_algo]['davies_bouldin']:.4f}
    
    ВАЖНОСТЬ ПРИЗНАКОВ:
    """)
    
    if importance is not None:
        for _, row in importance.iterrows():
            print(f"     {row['feature']}: {row['importance']:.3f}")
    
    lg.log_event("INFO", f"Segmentation completed. Best algorithm: {best_algo}")
    
    return seg, results, metrics, importance