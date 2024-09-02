import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def Kmeans(X, n):
    """
    进行 K-Means 聚类并返回聚类标签和聚类中心。
    参数:
        X: 数据点
        n: 聚类中心数目
    """
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    return y_kmeans, centers

# 生成和聚类数据集，并进行可视化
def generate_and_cluster_data(make_function, samples=400, centers=2, noise=0.05, cluster_std=1.0, **kwargs):
    # 根据函数类型调整参数
    if make_function.__name__ == 'make_blobs':
        X, y = make_function(n_samples=samples, centers=centers, cluster_std=cluster_std, **kwargs)
    else:
        X, y = make_function(n_samples=samples, noise=noise, **kwargs)
    
    y_kmeans, centers = Kmeans(X, centers)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='Spectral', edgecolor='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='yellow', s=200, alpha=0.5, marker='*')
    plt.title(f"{make_function.__name__} data clustered with K-Means")
    plt.show()

    # 评估聚类结果
    evaluate_clustering(y, y_kmeans)


# 聚类效果评估函数
def evaluate_clustering(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print("Accuracy Score:", acc)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Adjusted Rand Index (ARI):", ari)

# 对 make_circles, make_moons, make_blobs 分别生成和聚类数据
generate_and_cluster_data(make_circles, factor=0.5)
generate_and_cluster_data(make_moons)
generate_and_cluster_data(make_blobs, centers=3, cluster_std=1.5)

