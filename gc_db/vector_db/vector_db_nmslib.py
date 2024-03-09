import logging
import pickle
import time

import hnswlib
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBNMS:

    def __init__(self):
        self.db_vector = hnswlib.Index(space='cosine', dim=512)
        self.db_vector.init_index(max_elements=105100, ef_construction=200, M=16)

    def insert(self, vectors: list[np.array], vector_ids: list[int]):
        self.db_vector.add_items(vectors, vector_ids)

    def index_db(self):
        self.db_vector.set_ef(90)

    def query(self, query_vector: np.array, k: int = 20):
        results = self.db_vector.knn_query(query_vector, k=k)
        results = list(zip(results[0][0], results[1][0]))
        results.sort(key=lambda x: -x[1])
        return results


if __name__ == "__main__":
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open("../../data/dict_ids_embeddings_full.pickle", "rb"))
    VDB_IM = VectorDBNMS()
    if hasattr(VDB_IM, "insert"):
        start = time.time()
        to_insert = list(dict_ids_embeddings.values())
        to_insert_ids = list(dict_ids_embeddings.keys())
        VDB_IM.insert(to_insert, to_insert_ids)
        end = time.time()
        lasted = np.round(end - start, 3)
        # logger.info("Loading vector to memory db : " + str(len(VDB_IM.db_vector)))
        logger.info("Time elapsed to insert and index:" + str(lasted))

    if hasattr(VDB_IM, "query"):
        FCLIP = FashionCLIP('fashion-clip')
        embeded_query = FCLIP.encode_text(["White tee shirt with NASA logo"], 1)[0]
        logger.info("QUERY SHAPE:" + str(embeded_query.shape))
        start = time.time()
        nn = VDB_IM.query(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 6)
        logger.info("Time elapsed with exhaustive search:" + str(lasted))
        logger.info("Results : " + str(nn))
