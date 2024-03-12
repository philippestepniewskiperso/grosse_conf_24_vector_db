import logging
import pickle
import time

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from sklearn.cluster import KMeans

from gc_db.config import DICT_IDS_EMBEDDINGS_FULL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB_IM():

    def __init__(self):
        self.inverted_index: dict[int, list[np.array]] = {}
        self.codebook: list[np.array] = []
        self.db_vector: dict[int, np.array] = {}

    def insert(self, vector: np.array, vector_id: int):
        self.db_vector[vector_id] = vector

    def query(self, query_vector: np.array, k: int = 20):
        distances_and_ids: list[tuple[int, float]] = [(vector_id, self.cosine_similarity(query_vector, vector)) for
                                                      vector_id, vector in self.db_vector.items()]
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[0:k]

    def init_kmeans_index(self, nb_clusters: int):
        kmeans = KMeans(n_clusters=nb_clusters)
        kmeans.fit(list(self.db_vector.values()))
        self.codebook = kmeans.cluster_centers_
        for vector_id, vector in self.db_vector.items():
            predicted_cluster_id: int = kmeans.predict([vector])[0]
            if predicted_cluster_id in self.inverted_index:
                self.inverted_index[predicted_cluster_id].append(vector_id)
            else:
                self.inverted_index[predicted_cluster_id] = [vector_id]

    def query_with_kmeans(self, query_vector: np.array, k: int = 20, n_probes: int = 1) -> list[
        tuple[int, float]]:
        # trouver le centroide le plus proche du vecteur requête
        centroides_distances = [(centroide_id, self.cosine_similarity(query_vector, centroide)) for
                                centroide_id, centroide in enumerate(self.codebook)]
        centroides_distances.sort(key=lambda x: -x[1])
        np_centroide = centroides_distances[0]
        ##On récupère la liste des vecteurs dans ce centroide
        np_centroide_vectors = self.inverted_index[np_centroide[0]]
        ## On calcule les distances entre cette liste et le vecteur requête
        distances_and_ids = [(vector_id, self.cosine_similarity(self.db_vector[vector_id], query_vector)) for vector_id
                             in np_centroide_vectors]
        ## on trie et on retourne les k plus proches vecteurs
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[0:k]

    def cosine_similarity(self, vector_a: np.array, vector_b: np.array) -> float:
        num = np.dot(vector_a, vector_b)
        denom = np.dot(np.linalg.norm(vector_a), np.linalg.norm(vector_b))
        return num / denom


if __name__ == "__main__":
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open(DICT_IDS_EMBEDDINGS_FULL_PATH, "rb"))
    VDB_IM = VectorDB_IM()
    if hasattr(VDB_IM, "insert"):
        _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
        logger.info("Loading vector to memory db : " + str(len(VDB_IM.db_vector)))

    if hasattr(VDB_IM, "query"):
        FCLIP = FashionCLIP('fashion-clip')
        embeded_query = FCLIP.encode_text(["White tee shirt with NASA logo"], 1)[0]
        logger.info("QUERY SHAPE:" + str(embeded_query.shape))
        start = time.time()
        nn = VDB_IM.query(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with exhaustive search:" + str(lasted))
        logger.info("Results : " + str(nn))

    if hasattr(VDB_IM, 'init_kmeans_index'):
        start = time.time()
        VDB_IM.init_kmeans_index(nb_clusters=5)
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
