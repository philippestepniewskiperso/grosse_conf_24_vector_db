import pickle
from collections import defaultdict
import time

import nmslib
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import logging

from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBNMS:

    def __init__(self):
        self.db_vector = nmslib.init(method='hnsw', space='cosinesimil')

    def insert(self, vector: np.array, vector_id: int):
        self.db_vector.addDataPoint(vector_id,vector)
        self.db_vector.createIndex({'post': 2}, print_progress=True)



if __name__ == "__main__":
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open("../../data/dict_ids_embeddings_full.pickle", "rb"))
    VDB_IM = VectorDBNMS()
    if hasattr(VDB_IM, "insert"):
        start = time.time()
        _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Loading vector to memory db : " + str(len(VDB_IM.db_vector)))
        logger.info("Time elapsed to insert and index:" + str(lasted))

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
        logger.info("Computed hsnw index in " + str(lasted))
        logger.info("Codebook :" + str(VDB_IM.codebook)[:200])
        logger.info("Inverted index:" + str(VDB_IM.inverted_index)[:200])
        logger.info("Inverted index first cluster length: " + str(len(VDB_IM.inverted_index[0])))

    if hasattr(VDB_IM, 'query_with_kmeans'):
        start = time.time()
        nn = VDB_IM.query_with_kmeans(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with IVF search:" + str(lasted))
