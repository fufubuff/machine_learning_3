import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score
)
from scipy.optimize import linear_sum_assignment

def load_images(image_dir):
    """
    加载指定目录下的所有图像，并将其转换为一维数组格式的DataFrame。
    
    参数:
        image_dir (str): 存储图像的目录路径。

    返回:
        total_photo (DataFrame): 包含所有图像数据的DataFrame。
        target (list): 图像的目标标签列表。
    """
    total_photo = []
    target = []
    
    # 获取目录下所有子文件夹的名称
    subdirs = os.listdir(image_dir)
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(image_dir, subdir)
        if os.path.isdir(subdir_path):
            # 获取子文件夹内的所有图像文件
            for image_name in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, image_name)
                image = plt.imread(image_path)
                image = image.reshape(1, -1)
                total_photo.append(image)
                target.append(i)
    
    # 转换为DataFrame
    total_photo = pd.DataFrame(np.vstack(total_photo))
    return total_photo, target

def kmeans_clustering(data, n_clusters=10):
    """
    使用KMeans算法对数据进行聚类。

    参数:
        data (DataFrame): 待聚类的数据。
        n_clusters (int): 聚类的簇数量。

    返回:
        result (numpy.ndarray): 聚类后的图像数据。
        y_predict (numpy.ndarray): 聚类标签。
    """
    clf = KMeans(n_clusters=n_clusters, n_init='auto')
    clf.fit(data)
    y_predict = clf.predict(data)
    centers = clf.cluster_centers_
    
    # 根据聚类结果生成图像
    result = centers[y_predict]
    result = result.astype("int64").reshape(-1, 200, 180, 3)
    return result, y_predict

def plot_clusters(images, nrows=10, ncols=20):
    """
    可视化聚类后的图像结果。

    参数:
        images (numpy.ndarray): 聚类后的图像数据。
        nrows (int): 子图的行数。
        ncols (int): 子图的列数。
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(15, 8), dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            if count < len(images):
                ax[i, j].imshow(images[count])
                ax[i, j].axis('off')  # 隐藏坐标轴
                count += 1
    plt.show()

def evaluate_clustering(true_labels, predicted_labels):
    """
    评价聚类效果，输出相关评价指标。

    参数:
        true_labels (list): 真实的标签。
        predicted_labels (numpy.ndarray): 预测的聚类标签。
    """
    acc = accuracy_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    print(f"ACC = {acc}")
    print(f"NMI = {nmi}")
    print(f"ARI = {ari}")

if __name__ == '__main__':
    # 设置图像目录路径
    image_dir = r'face_images'
    
    # 加载图像数据
    total_photo, target = load_images(image_dir)
    
    # 进行KMeans聚类
    result, y_predict = kmeans_clustering(total_photo, n_clusters=10)
    
    # 评价聚类效果
    evaluate_clustering(target, y_predict)
    
    # 可视化聚类结果
    plot_clusters(result)
