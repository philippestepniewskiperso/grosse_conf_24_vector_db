import numpy as np
from sklearn.cluster import KMeans

class VectorDB_IM:

    # initialisation de la base de données
    def __init__(self):
        self.inverted_index: dict[int, list[np.array]] = {}
        self.codebook: list[np.array] = []
        self.db_vector: dict[int, np.array] = {}

    def insert(self, vector: np.array, vector_id: int):
        # on insert le vecteur d'embedding avec son id en clé
        self.db_vector[vector_id] = vector

    def query(self, query_vector: np.array, k: int = 20) -> list[tuple[int, float]]:
        # on veut retourner une liste au format [(vector_id,similarité),()..] = [(1432,0.4),(12,0.39)]
        distances_and_ids: list[tuple[int, float]] = [
            (vector_id, self.cosine_similarity(query_vector, self.db_vector[vector_id])) for vector_id in
            self.db_vector.keys()]
        # on trie la liste par similarité décroissante
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[:k]

    def init_kmeans_index(self, nb_clusters: int = 10):
        # initialisons un modèle kmeans de nb_clusters
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        # on le fit sur tous les vecteurs de la base
        kmeans.fit(list(self.db_vector.values()))
        # on stocke les centroides dans le codebook
        self.codebook: list[np.array] = kmeans.cluster_centers_
        # maintenant stockons chaque cluster_id avec la liste de ses vecteurs {4:[423,3432],2:[343,1]...
        self.inverted_index: dict[int, list[np.array]] = {}
        for vector_id in self.db_vector.keys():
            cluster_id = kmeans.predict([self.db_vector[vector_id]])[0]
            if cluster_id not in self.inverted_index:
                self.inverted_index[cluster_id] = [vector_id]
            else:
                self.inverted_index[cluster_id].append(vector_id)

    def query_with_kmeans(self, query_vector: np.array, n_probes: int = 1, k: int = 20) -> list[tuple[int, float]]:
        # récupérer le centroide_id le plus proche du vecteur requête
        # centroids = [[],[],[]] sans clé, l'index est l'id
        distance_query_centroids: list[tuple[int, float]] = [
            (cluster_id, self.cosine_similarity(query_vector, centroid)) for cluster_id, centroid in
            enumerate(self.codebook)]
        distance_query_centroids.sort(key=lambda x: -x[1])
        nearest_cluster_ids = distance_query_centroids[:n_probes]
        print("NB CLUSTERS SELECTED =", len(nearest_cluster_ids))
        # on récupère la liste des vecteurs des n_probes plus proches clusters
        n_probes_vector_ids: list[int] = []
        for cluster_id, _ in nearest_cluster_ids:
            n_probes_vector_ids = n_probes_vector_ids + self.inverted_index[cluster_id]
        # on calcul la similarité avec tous ces vecteurs  et la requête
        distances_and_ids: list[tuple[int, float]] = [
            (vector_id, self.cosine_similarity(query_vector, self.db_vector[vector_id])) for vector_id in
            n_probes_vector_ids]
        # on trie
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[:k]

    def cosine_similarity(self, vector_a: np.array, vector_b: np.array):
        # produit scalair de a et b
        num = np.dot(vector_a, vector_b)
        # produit scalaire des normes de a et b
        denom = np.dot(np.linalg.norm(vector_a), np.linalg.norm(vector_b))
        return num / denom


if __name__ == "__main__":
    import logging
    import pickle
    import time
    from gc_db.config import DICT_IDS_EMBEDDINGS_PATH
    from fashion_clip.fashion_clip import FashionCLIP
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open(DICT_IDS_EMBEDDINGS_PATH, "rb"))
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
