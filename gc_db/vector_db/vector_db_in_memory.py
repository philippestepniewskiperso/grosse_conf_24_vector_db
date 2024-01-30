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
        self.codebook: np.array = None
        self.inverted_index: dict[int, list[int]] = defaultdict(list)
        self.external_ids_list: list[int] = []
        self.db_vector_list: list[np.array] = []

    def insert(self, vector: np.array, external_id: int):
        self.db_vector_list.append(vector)
        self.external_ids_list.append(external_id)

    def query(self, query_vector: np.array, k: int = 20) -> list[tuple[int, float]]:
        # calcul des similarités avec tous vecteurs
        similarities: list[float] = [self.cosine_similarity(query_vector, vector) for vector in self.db_vector_list]
        k_idx_sim = np.argpartition(similarities, -k)[-k:]
        # liste des tuples de ids externe et leur similarités
        k_external_ids_and_sim: list[tuple[int, float]] = [(self.external_ids_list[idx], similarities[idx]) for idx in
                                                           k_idx_sim]
        k_external_ids_and_sim.sort(key=lambda x: -x[1])
        return k_external_ids_and_sim

    def init_kmeans_index(self, nb_clusters: int = 10):
        kmeans = KMeans(n_clusters=nb_clusters)
        kmeans.fit(self.db_vector_list)

        if len(self.inverted_index) > 0:
            self.inverted_index = {}
        self.codebook = kmeans.cluster_centers_
        predicted_cluster_ids: np.array = kmeans.predict(self.db_vector_list)

        for vector_idx, cluster_id in enumerate(predicted_cluster_ids):
                self.inverted_index[cluster_id].append(vector_idx)

    def query_with_kmeans(self, query_vector: np.array, n_probes: int = 1, k: int = 20) -> list[tuple[int, float]]:
        # calculer les similarités entre query vector et tous les centroides
        centroids_similarities: list[float] = [self.cosine_similarity(query_vector, centroid) for centroid in
                                               self.codebook]
        n_probes_centroids_idx = np.argpartition(centroids_similarities, -n_probes)[-n_probes:]
        # récupérer tous les vecteurs
        list_vectors_idx_n_probes = [vector_idx for centroid_idx in n_probes_centroids_idx for vector_idx in
                                     self.inverted_index[centroid_idx]]
        # Similarités entre le vecteur requête et tous les vecteurs appartenant aux clusters les plus proches
        similarities = [self.cosine_similarity(query_vector, self.db_vector_list[vector_idx]) for vector_idx in
                        list_vectors_idx_n_probes]
        k_vectors_idx_local = np.argpartition(similarities, -k)[-k:]
        external_vector_ids_and_sim = [
            (self.external_ids_list[list_vectors_idx_n_probes[local_vector_idx]], similarities[local_vector_idx]) for
            local_vector_idx in k_vectors_idx_local]
        external_vector_ids_and_sim.sort(key=lambda x: -x[1])
        return external_vector_ids_and_sim

    def cosine_similarity(self, query_vector, vector):
        num = np.dot(query_vector, vector)
        denom = np.dot(np.linalg.norm(query_vector), np.linalg.norm(vector))
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
