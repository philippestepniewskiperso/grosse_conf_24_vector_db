import streamlit as st
import gc_db.streamlit.st_creators as stc
from gc_db.utils.utils import get_path_from_image_id
from gc_db.vector_db.vector_db_in_memory import VectorDB_IM
import pickle
import numpy as np
import time
from fashion_clip.fashion_clip import FashionCLIP

if "is_initiated" not in st.session_state:
    print("INITIATING")
    dict_ids_embeddings = pickle.load(open("data/dict_ids_embeddings.pickle", "rb"))
    VDB_IM = VectorDB_IM()
    # logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    VDB_IM.init_kmeans_index()
    st.session_state["VDB_IM"] = VDB_IM
    st.session_state["FCLIP"] = FashionCLIP('fashion-clip')
    st.session_state["is_initiated"] = True
else:
    VDB_IM = st.session_state["VDB_IM"]
    FCLIP = st.session_state["FCLIP"]

st.set_page_config(layout="wide")
st.title("MOTEUR DE RECHERCHE MULTIMODAL")
with st.sidebar:
    st.image("./data/assets/gc_logo.webp")

col1, col2 = st.columns(2)
with col1:
    st.markdown("## Votre requête:")
with col2:
    query_text = st.text_input("")
if query_text != "":
    embeded_query = FCLIP.encode_text([query_text], 1)[0]
    start = time.time()
    nn = VDB_IM.query_with_kmeans(embeded_query,n_probes=10)
    end = time.time()
    lasted = np.round(end - start, 3)
    #   logger.info(f"Results in {lasted} for first vector in dict as query" + str(nn))

    #  logger.debug("images: " + str(nn[:25]))
    print("images: " + str(nn[:25]))
    image_pathes = [get_path_from_image_id(dist_id[0]) for dist_id in nn]
    st.info(f"Temps d'éxecution de la requête : **{lasted} seconds**")
    stc.display_result_gallery(image_pathes, 5)
