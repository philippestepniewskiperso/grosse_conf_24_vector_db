import logging
import pickle
import time

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

from gc_db.config import DICT_IDS_EMBEDDINGS_FULL_PATH
from gc_db.streamlit.st_utils import load_logs_of_the_day_if_exists
from gc_db.utils.image_segmentation import ClothSegmenter
from gc_db.vector_db.vector_db_nmslib import VectorDBNMS
from gc_db.vector_db.vector_db_in_memory import VectorDB_IM

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vdb_im(dict_ids_embeddings: dict[int, np.array], init_kmeans=True):
    VDB_IM = VectorDB_IM()
    logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    if hasattr(VDB_IM, "init_kmeans_index") and init_kmeans:
        st.session_state["n_clusters"] = 10
        start = time.time()
        VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
        stop = time.time()
        logger.info(f"Kmeans indexed in {str(stop - start)}")
    return VDB_IM


def init_streamlit(hsnw: bool):
    load_logs_of_the_day_if_exists()
    if "is_initiated" not in st.session_state:
        logger.info("INITIATING")
        dict_ids_embeddings = pickle.load(open(DICT_IDS_EMBEDDINGS_FULL_PATH, "rb"))
        VDB_IM = None
        VDB_NMS = None
        if not hsnw:
            VDB_IM = init_vdb_im(dict_ids_embeddings)
        else:
            # used to compute knn recall
            VDB_IM = init_vdb_im(dict_ids_embeddings, init_kmeans=False)
            VDB_NMS = VectorDBNMS()
            logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
            to_insert = list(dict_ids_embeddings.values())
            to_insert_ids = list(dict_ids_embeddings.keys())
            VDB_NMS.insert(to_insert, to_insert_ids)

        st.session_state["VDB_IM"] = VDB_IM
        st.session_state["VDB_NMS"] = VDB_NMS
        FCLIP = FashionCLIP('fashion-clip')
        SEG = ClothSegmenter()
        st.session_state["FCLIP"] = FCLIP
        st.session_state["SEG"] = SEG
        st.session_state["is_initiated"] = True
    else:
        VDB_IM = st.session_state["VDB_IM"]
        FCLIP = st.session_state["FCLIP"]
        SEG = st.session_state["SEG"]

    return VDB_IM, FCLIP, SEG
