import pickle
import time

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from sklearn.cluster import KMeans

from gc_db.vector_db.vector_db import VectorDB


class VectorDB_IM(VectorDB):

    def __init__(self):
        self.codebook = None
        self.db_vector_list = []
        self.inverted_index = {}
        self.db_external_ids = []

    def insert(self, vector: np.array, external_id: int):
        self.db_vector_list.append(vector)
        self.db_external_ids.append(external_id)

    def query(self, query_vector: np.array, k: int = 10):
        distances = [self.cosine_similarity(query_vector, vector) for vector in self.db_vector_list]
        nearest_ids = np.argpartition(distances, -k)[-k:]
        nearest_ext_ids_and_dist = [(self.db_external_ids[internal_id], distances[internal_id]) for internal_id in
                                    nearest_ids]
        nearest_ext_ids_and_dist.sort(key=lambda x: -x[1])
        return nearest_ext_ids_and_dist

    def query_with_kmeans(self, query_vector: np.array, k: int = 10, n_probes: int = 2):
        distances_with_centroids = [self.cosine_similarity(query_vector, centroid) for centroid in self.codebook]
        nearest_centroids_ids = np.argpartition(distances_with_centroids, -n_probes)[-n_probes:]
        vectors_ids_in_probes = [vector_id for centroid_id in nearest_centroids_ids for vector_id in
                                 self.inverted_index[centroid_id]]
        distances = [self.cosine_similarity(query_vector, self.db_vector_list[vector_id]) for vector_id in
                     vectors_ids_in_probes]
        nearest_vectors_index = np.argpartition(distances, -k)[-k:]
        nearest_vectors_ids_and_distance = [
            (self.db_external_ids[vectors_ids_in_probes[vector_index]], distances[vector_index]) for
            vector_index in nearest_vectors_index]
        nearest_vectors_ids_and_distance.sort(key=lambda x: -x[1])
        print(nearest_vectors_ids_and_distance)
        return nearest_vectors_ids_and_distance

    def init_kmeans_index(self, nb_clusters: int = 10):
        kmeans = KMeans(n_clusters=nb_clusters)
        kmeans.fit(self.db_vector_list)
        self.codebook = kmeans.cluster_centers_
        predicted_clusters = kmeans.predict(self.db_vector_list)
        for internal_id, cluster_id in enumerate(predicted_clusters):
            if cluster_id in self.inverted_index:
                current_list = self.inverted_index[cluster_id]
                current_list.append(internal_id)
            else:
                self.inverted_index[cluster_id] = [internal_id]

    def cosine_similarity(self, query_vector, vector):
        num = np.dot(query_vector, vector)
        denom = np.dot(np.linalg.norm(query_vector), np.linalg.norm(vector))
        return num / denom


if __name__ == "__main__":
    dict_ids_embeddings = pickle.load(open("../../data/dict_ids_embeddings.pickle", "rb"))
    VDB_IM = VectorDB_IM()
    # logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    VDB_IM.init_kmeans_index()
    FCLIP = FashionCLIP('fashion-clip')
    embeded_query = FCLIP.encode_text(["White tee shirt with NASA logo"], 1)[0]
    start = time.time()
    nn = VDB_IM.query_with_kmeans(embeded_query)
    end = time.time()
    lasted = np.round(end - start, 3)
    print("Time elapsed:" + str(lasted))
