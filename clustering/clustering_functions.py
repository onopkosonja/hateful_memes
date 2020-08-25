from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Image folder

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image_folder = ImageFolder('~/Desktop/data/train/', transform=data_transform)
DATALOADER = DataLoader(image_folder, batch_size=32)


# Plot T-SNE where labels are from clustering

def tsne(embs, labels, model_name, metric='euclidean'):
    tsne = TSNE(metric=metric)
    X = tsne.fit_transform(embs)
    plt.title('T-SNE for {}'.format(model_name), fontsize=25)
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels, legend='full',
                    palette=sns.color_palette("hls", len(np.unique(labels))))


# Print sample images for several clusters

def plot_samples(labels, n_clusters=3, n_imgs=5):
    clusters = np.unique(labels[labels != -1])
    num_cluster = list(np.random.choice(clusters, size=n_clusters, replace=False))
    num_cluster.append(-1)

    fig = plt.figure(figsize=(40, 30))
    for i, cl in enumerate(num_cluster):
        imgs = np.where(labels == cl)[0]

        if len(imgs) >= n_imgs:
            samples = np.random.choice(imgs, size=n_imgs, replace=False)

            for j, idx in enumerate(samples):
                img_path, _ = DATALOADER.dataset.samples[idx]
                plt.subplot(n_clusters + 1, n_imgs, i * n_imgs + j + 1)
                if cl == -1:
                    plt.title('Outlier', fontsize=40)
                else:
                    plt.title('Cluster {}'.format(cl), fontsize=40)
                img = plt.imread(img_path)
                plt.imshow(img)
    plt.show()


# K-means

def k_means(embeddings, n_clusters, return_labels=True):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)

    if return_labels:
        labels = kmeans.fit_predict(embeddings)
        return labels

    distances = kmeans.fit_transform(embeddings)
    closest_distances = np.min(distances, axis=1)
    return np.mean(closest_distances)

# Plot number of clusters vs average distance from centroids

def plot_kmeans_params(embeddings, n_clusters_list, title=''):
    mean_distances = []
    for n_clusters in tqdm(n_clusters_list):
        dist = k_means(embeddings, n_clusters, return_labels=False)
        mean_distances.append(dist)
    plt.plot(n_clusters_list, mean_distances, label=title)


# DBSCAN

def dbscan(embeddings, eps, min_samples, metric='euclidean', return_labels=True):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(embeddings)
    labels = dbscan.labels_
    if return_labels:
        return labels

    n_clusters = len(np.unique(labels)) - 1
    n_noisy = len(np.where(labels == -1)[0])
    return n_clusters, n_noisy


def get_dist_list(X, metric, n=5):
    if metric == 'euclidean':
        dist_matr = euclidean_distances(X)
        min_dist = np.min(dist_matr[dist_matr != 0])
        max_dist = np.max(dist_matr) // 2
    else:
        min_dist, max_dist = 1e-5, 1
    step = (max_dist - min_dist) / n
    return np.arange(min_dist, max_dist, step)


""" Plot eps(maximum distance between two samples for one to be considered as in the neighborhood of the other) vs
 number of clusters and eps vs min_samples(The number of samples in a neighborhood for a point 
 to be considered as a core point)"""


def dbscan_eps_with_samples(embeddings, eps_list, min_samples, plot_1, plot_2, metric='euclidean'):
    params = []
    for eps in tqdm(eps_list):
        p = dbscan(embeddings, eps, min_samples, metric=metric, return_labels=False)
        params.append(p)
    clusters_eps, noisy_eps = list(zip(*params))
    plot_1.plot(eps_list, np.array(clusters_eps), label=str(min_samples) + ' neighbors')
    plot_2.plot(eps_list, np.array(noisy_eps), label=str(min_samples) + ' neighbors')


def plot_dbscan_params(embeddings, min_samples_list, metric='euclidean'):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    eps_list = get_dist_list(embeddings, metric)
    for s in min_samples_list:
        dbscan_eps_with_samples(embeddings, eps_list, s, ax1, ax2, metric)

    ax1.set_xlabel('Maximum distance between neighbors', fontsize=15)
    ax2.set_xlabel('Maximum distance between neighbors', fontsize=15)
    ax1.set_ylabel('Number of clusters', fontsize=15)
    ax2.set_ylabel('Number of noise samples', fontsize=15)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.show()



