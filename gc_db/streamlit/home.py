import os

import streamlit as st
import gc_db.streamlit.st_creators as stc
from gc_db.utils.flex_logging import stream_handler
from gc_db.utils.utils import get_path_from_image_id
from gc_db.vector_db.vector_db_in_memory import VectorDB_IM
import logging
import pickle
import numpy as np
import time
from fashion_clip.fashion_clip import FashionCLIP

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

if "vector_db" not in st.session_state.keys():
    dict_ids_embeddings = pickle.load(open("data/dict_ids_embeddings.pickle", "rb"))
    VECTOR_DB = VectorDB_IM()
    logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VECTOR_DB.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    VECTOR_DB.compute_kmeans_clustering()
    FCLIP = FashionCLIP('fashion-clip')
    st.session_state["vector_db"] = VECTOR_DB
    st.session_state["fclip"] = FCLIP
else:
    VECTOR_DB = st.session_state["vector_db"]
    FCLIP = st.session_state["fclip"]

st.set_page_config(layout="wide")
st.title("Moteur de recherche texte/image")

with st.sidebar:
    query_text = st.text_input("Que recherchez vous?")
    stc.prepare_check_boxes()
if query_text != "":
    embeded_query = FCLIP.encode_text([query_text], 1)[0]
    start = time.time()
    if st.session_state["use_kmeans_index"]:
        nn = VECTOR_DB.query_with_kmeans_index(embeded_query)
    else:
        nn = VECTOR_DB.query(embeded_query)

    end = time.time()
    lasted = np.round(end - start, 4)
    logger.info(f"Results in {lasted} for first vector in dict as query" + str(nn))

    logger.debug("images: " + str(nn[:25]))
    image_pathes = [get_path_from_image_id(dist_id[1]) for dist_id in nn]
    st.write(f"Temps d'éxecution de la requête :{lasted} seconds")
    stc.display_result_gallery(image_pathes, 5)
