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

dict_ids_embeddings = pickle.load(open("data/dict_ids_embeddings_full.pickle", "rb"))
VDB_IM = VectorDB_IM()
logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
_ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
FCLIP = FashionCLIP('fashion-clip')

st.set_page_config(layout="wide")
st.title("Moteur de recherche texte/image")
with st.sidebar:
    query_text = st.text_input("Que recherchez vous?")
if query_text != "":
    embeded_query = FCLIP.encode_text([query_text], 1)[0]
    start = time.time()
    nn = VDB_IM.query(embeded_query)
    end = time.time()
    lasted = np.round(end - start,4)
    logger.info(f"Results in {lasted} for first vector in dict as query" + str(nn))

    logger.debug("images: " + str(nn[:25]))
    image_pathes = [get_path_from_image_id(dist_id[1]) for dist_id in nn]
    st.write(f"Temps d'éxecution de la requête :{lasted} seconds")
    stc.display_result_gallery(image_pathes, 5)
