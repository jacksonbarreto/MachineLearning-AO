import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

from kmeans.kmeans import KMeans

if __name__ == '__main__':
    dataframe = pd.read_csv("heart.csv", header=0)
    df = dataframe
    print(df.head())
    x_std = StandardScaler().fit_transform(df)
    print(x_std[1, 0])

    km = KMeans(n_clusters=2, random_state=42)
    km.fit(x_std)
    centroids = km.centroids

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(x_std[km.labels == 0, 0], x_std[km.labels == 0, 1],
                c='green', label='cluster 1')
    plt.scatter(x_std[km.labels == 1, 0], x_std[km.labels == 1, 1],
                c='blue', label='cluster 2')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
                c='r', label='centroid')
    plt.legend()
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    #plt.xlabel('Eruption time in mins')
    #plt.ylabel('Waiting time to next eruption')
    #plt.title('Visualization of clustered data', fontweight='bold')
    ax.set_aspect('equal')
    plt.show()

    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(x_std)
        sse.append(km.error)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    #plt.show()
