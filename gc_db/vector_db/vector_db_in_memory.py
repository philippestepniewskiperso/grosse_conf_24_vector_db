import pickle
from collections import defaultdict

import numpy as np
import time
from fashion_clip.fashion_clip import FashionCLIP
import logging

from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB_IM():
    def __init__(self):
        self.inverted_index: dict[int, list[id]] = {}
        self.codebook: list[np.array] = []
        self.db_vector_list: dict[int, np.array] = {}

    def insert(self, vector: np.array, external_id: int):
        self.db_vector_list[external_id] = vector

    def query(self, query_vector: np.array, k: int = 20) -> list[tuple[int, float]]:
        distances_and_ids = [(vector_id, self.cosine_similarity(query_vector, self.db_vector_list[vector_id])) for
                             vector_id in
                             self.db_vector_list.keys()]
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[:k]

    def init_kmeans_index(self, nb_clusters: int = 10):
        kmeans = KMeans(n_clusters=nb_clusters)
        kmeans.fit(list(self.db_vector_list.values()))
        self.codebook = kmeans.cluster_centers_
        for vector_id in self.db_vector_list.keys():
            predicted_cluster_id = kmeans.predict([self.db_vector_list[vector_id]])[0]
            if predicted_cluster_id in self.inverted_index:
                self.inverted_index[predicted_cluster_id].append(vector_id)
            else:
                self.inverted_index[predicted_cluster_id] = [vector_id]

    def query_with_kmeans(self, query_vector: np.array, n_probes: int = 1, k: int = 20) -> list[tuple[int, float]]:
        # on récupère le cluster le plus proche du vecteur requête
        distances_vector_clusters: list[tuple[int, float]] = [
            (centroid_id, self.cosine_similarity(query_vector, centroid)) for
            centroid_id, centroid in enumerate(self.codebook)]
        distances_vector_clusters.sort(key=lambda x: -x[1])
        nearest_centroid_ids = [tuple_id_dist[0] for tuple_id_dist in distances_vector_clusters[0:n_probes]]
        vectors_in_nearest_cluster = [vector for centroid_id in nearest_centroid_ids for vector in
                                      self.inverted_index[centroid_id]]
        distances_vectors_and_ids = [(vector_id, self.cosine_similarity(self.db_vector_list[vector_id], query_vector))
                                     for vector_id in
                                     vectors_in_nearest_cluster]
        distances_vectors_and_ids.sort(key=lambda x: -x[1])
        return distances_vectors_and_ids[0:k]

    def cosine_similarity(self, vector_a: np.array, vector_b: np.array):
        num = np.dot(vector_a, vector_b)
        denom = np.dot(np.linalg.norm(vector_a), np.linalg.norm(vector_b))
        return num / denom


if __name__ == "__main__":
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open("../../data/dict_ids_embeddings_full.pickle", "rb"))
    VDB_IM = VectorDB_IM()
    if hasattr(VDB_IM, "insert"):
        _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
        logger.info("Loading vector to memory db : " + str(len(VDB_IM.db_vector_list)))

    if hasattr(VDB_IM, "query"):
        FCLIP = FashionCLIP('fashion-clip')
        embeded_query = FCLIP.encode_text(["White tee shirt with NASA logo"], 1)[0]
        start = time.time()
        nn = VDB_IM.query(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with exhaustive search:" + str(lasted))
        logger.info("Results : " + str(nn))

    if hasattr(VDB_IM, 'init_kmeans_index'):
        start = time.time()
        VDB_IM.init_kmeans_index()
        end = time.time()
        lasted = np.round(end - start, 3)
        # 2 secondes sur le sample
        # 11 secondes sur le full
        logger.info("Computed kmeans index in " + str(lasted))
        logger.info("Codebook :" + str(VDB_IM.codebook)[:200])
        logger.info("Inverted index:" + str(VDB_IM.inverted_index)[:200])
        logger.info("Inverted index first cluster length: " + str(len(VDB_IM.inverted_index[0])))

    if hasattr(VDB_IM, 'query_with_kmeans'):
        start = time.time()
        nn = VDB_IM.query_with_kmeans(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with IVF search:" + str(lasted))
