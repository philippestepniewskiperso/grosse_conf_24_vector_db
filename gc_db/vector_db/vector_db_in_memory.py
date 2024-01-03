import numpy as np

from gc_db.vector_db import vector_db


class VectorDBIM(vector_db):

    def __init__(self):
        self.db_struct: dict[int, np.array] = {}

    def insert(self, vector: np.array, external_id: int):
        self.db_struct[external_id] = vector

    def query(self, query_vector: np.array, k: int = 20):
        distances_and_ids = [(external_id,cosine_similarity(query_vector,self.db_struct[external_id])) for external_id in self.db_struct.keys()]
        knn = np.argpartition(distances_and_ids,-k)[-k:]
        return knn

    def cosine_similarity(self,vector_A:np.array,vector_B:np.array):
        numerateur = np.dot(vector_A,vector_B)
        denominateur = np.
        res = numerateur/denominateur