from sklearn.cluster import KMeans
import math


def cluster_senteces(sentece_vectors):
    number_of_clusters = math.floor(math.sqrt(len(sentece_vectors)))
    kmeans = KMeans(number_of_clusters,random_state=0,n_init=10).fit(sentece_vectors)
    return kmeans