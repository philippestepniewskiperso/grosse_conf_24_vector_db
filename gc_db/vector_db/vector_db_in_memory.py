import numpy as np
from sklearn.cluster import KMeans

from gc_db.vector_db.vector_db import VectorDB


class VectorDB_IM(VectorDB):

    def __init__(self):
        self.vector_list: list[np.array] = []
        self.ids_list: list[int] = []
        self.kmeans_index: dict[int, list[int]] = {}

    def insert(self, vector: np.array, external_id: int):
        self.vector_list.append(vector)
        self.ids_list.append(external_id)

    def query(self, query_vector: np.array, k: int = 20):
        distances = [self.cosine_similarity(vector, query_vector) for vector in self.vector_list]
        knn = np.argpartition(distances, -k)[-k:]
        knn_ids_and_distances = [(self.ids_list[id], distances[id]) for id in knn]
        knn_ids_and_distances.sort(key=lambda x: -x[1])
        return knn_ids_and_distances

    def create_kmeans_index(self, n_clusters: int = 20):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.vector_list)
        predicted_clusters = kmeans.predict(self.vector_list)
        for vector_id, predicted_cluster in enumerate(predicted_clusters):
            if predicted_cluster in self.kmeans_index:
                old_list = self.kmeans_index[predicted_cluster]
                old_list.append(vector_id)
                self.kmeans_index[predicted_cluster] = old_list
            else:
                self.kmeans_index[predicted_cluster] = [vector_id]
        self.codebook = kmeans.cluster_centers_

    def query_with_kmeans_index(self, query_vector: np.array, k: int = 20, n_probes: int = 1):
        centers_distance = [self.cosine_similarity(center, query_vector) for center in self.codebook]
        nearest_centers_ids = np.argpartition(centers_distance, -n_probes)[-n_probes:]
        vector_ids_probe_list = [vector for cluster_id in nearest_centers_ids for vector in
                                 self.kmeans_index[cluster_id]]
        distances = [self.cosine_similarity(self.vector_list[vector_id], query_vector) for vector_id in
                     vector_ids_probe_list]
        knn_vector_ids = np.argpartition(distances, -k)[-k:]
        knn_vector_local_ids = [vector_ids_probe_list[local_id] for local_id in knn_vector_ids]
        knn_vector_external_ids = [self.ids_list[id] for id in knn_vector_local_ids]
        knn_distances = [distances[local_id] for local_id in knn_vector_ids]
        external_ids_and_distance = list(zip(knn_vector_external_ids, knn_distances))
        external_ids_and_distance.sort(key=lambda x: -x[1])
        return external_ids_and_distance

    @staticmethod
    def cosine_similarity(vector, query_vector):
        num = np.dot(vector, query_vector)
        denom = np.dot(np.linalg.norm(vector), np.linalg.norm(vector))
        return num / denom
