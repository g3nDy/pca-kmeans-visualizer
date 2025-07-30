import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generate_points(func, num_points=50):
    points = []
    for i in range(num_points):
        points.append(func(i))
    return points

def rule_1(i):
    x0 = i * 2
    x1 = randint(-10, 73) * np.arctan(i)
    x2 = randint(1, 10)
    x3 = randint(-5, 15)
    x4 = randint(5, 20)
    return [x0, x1, x2, x3, x4]

def rule_2(i):
    x0 = np.pi * randint(52, 73)
    x1 = i * 2 + 40
    x2 = i * 5 + 20 + randint(-5, 10)
    x3 = i * 2 + 20
    x4 = i + 12
    return [x0, x1, x2, x3, x4]

def rule_3(i):
    x0 = i * 3 + 50
    x1 = np.sin(i) * 2
    x2 = np.tan(i ** 2) + randint(-1, 10)
    x3 = 1
    x4 = -17 / (i + 1)
    return [x0, x1, x2, x3, x4]



def pca_and_clustering(n_components, all_points):
    # Применение PCA
    pca = PCA(n_components)
    reduced_points = pca.fit_transform(all_points)
    #кластеризация 
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(reduced_points)

    return reduced_points, labels


def create_graph(n_components, reduced_points, labels):
    # Создание графика
    fig = plt.figure(figsize=(10, 10))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(reduced_points[:, 0], reduced_points[:, 1], c=labels)

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_points[:, 0], reduced_points[:, 1], reduced_points[:, 2], c=labels)
    plt.show()



num_points = 50
all_points = np.array(generate_points(rule_1, num_points) + generate_points(rule_2, num_points) + generate_points(rule_3, num_points))

    # Выбор количества компонентов после уменьшения размерности
n_components = 3
reduced_points, labels = pca_and_clustering(n_components, all_points)

    # Рисование графика
create_graph(n_components, reduced_points, labels)
