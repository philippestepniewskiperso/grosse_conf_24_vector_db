import numpy as np

from gc_db.vector_db.vector_db import VectorDB
from sklearn.cluster import KMeans


class VectorDB_IM(VectorDB):

    def __init__(self):
        self.db_struct: dict[int, np.array] = {}
        self.kmeans_index: dict[int, list[int]] = {}
        self.centroids: dict[int, np.array] = {}

    def insert(self, vector: np.array, external_id: int):
        self.db_struct[external_id] = vector

    def query(self, query_vector: np.array, k: int = 20):
        distances_and_ids = [(external_id, self.cosine_similarity(query_vector, self.db_struct[external_id])) for
                             external_id in self.db_struct.keys()]
        external_ids, distances = zip(*distances_and_ids)
        knn_ids = np.argpartition(distances, -k)[-k:]
        nn_distances_external_ids = [(distances[id], external_ids[id]) for id in knn_ids]
        nn_distances_external_ids.sort(key=lambda x:-x[0])
        return nn_distances_external_ids

    @staticmethod
    def cosine_similarity(vector_A: np.array, vector_B: np.array):
        numerateur = np.dot(vector_A, vector_B)
        denominateur = np.dot(np.linalg.norm(vector_A), np.linalg.norm(vector_B))
        cos_sim = numerateur / denominateur
        return cos_sim

    def compute_kmeans_clustering(self, nb_clusters: int = 10):
        kmeans = KMeans(n_clusters=nb_clusters)
        X = [vector for vector in self.db_struct.values()]
        kmeans.fit(np.array(X))
        for external_id in self.db_struct.keys():
            vector = self.db_struct[external_id]
            cluster_id = kmeans.predict(np.array([vector]))[0]
            self.add_vector_to_kmeans_index(cluster_id, external_id)

        for centroid_id, centroid in enumerate(kmeans.cluster_centers_):
            self.centroids[centroid_id] = centroid

    def add_vector_to_kmeans_index(self, cluster_id: int, external_id: int):
        if cluster_id in self.kmeans_index.keys():
            cluster_vector_list = self.kmeans_index[cluster_id]
            cluster_vector_list.append(external_id)
            self.kmeans_index[cluster_id] = cluster_vector_list
        else:
            cluster_vector_list = [external_id]
            self.kmeans_index[cluster_id] = cluster_vector_list

    def find_knearest_clusters_to_vector(self, query_vector: np.array, n_probes):
        distances_and_cluster_ids = [(cluster_id, self.cosine_similarity(query_vector, self.centroids[cluster_id])) for
                                     cluster_id in
                                     self.centroids.keys()]
        cluster_ids, distances = zip(*distances_and_cluster_ids)
        nnclusters_ids = np.argpartition(distances, -n_probes)[-n_probes:]
        return nnclusters_ids

    def query_with_kmeans_index(self, query_vector: int, k: int = 20, n_probes: int = 1):
        nn_clusters_ids = self.find_knearest_clusters_to_vector(query_vector, n_probes=n_probes)
        all_distances = []
        for cluster_id in nn_clusters_ids:
            cluster_vector_ids_list = self.kmeans_index[cluster_id]
            for external_id in cluster_vector_ids_list:
                all_distances.append((external_id, self.cosine_similarity(query_vector, self.db_struct[external_id])))
        external_ids, distances = zip(*all_distances)
        nn_ids = np.argpartition(distances, -k)[-k:]
        nn_distances_external_ids = [(distances[id], external_ids[id]) for id in nn_ids]
        nn_distances_external_ids.sort(key=lambda x: -x[0])
        return nn_distances_external_ids
