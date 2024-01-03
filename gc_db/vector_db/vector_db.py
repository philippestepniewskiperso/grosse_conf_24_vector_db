from abc import ABC, abstractmethod
import numpy as np


class VectorDB(ABC):

    @abstractmethod
    def insert(self, vector: np.array, external_id: int): pass

    @abstractmethod
    def query(self, vector: np.array): pass
