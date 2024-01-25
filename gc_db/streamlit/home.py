import logging as logger
import pickle
import time

import streamlit as st
from fashion_clip.fashion_clip import FashionCLIP

from gc_db.streamlit.st_utils import perform_query
from gc_db.vector_db.vector_db_in_memory import VectorDB_IM

st.session_state["k"] = 20

if "is_initiated" not in st.session_state:
    print("INITIATING")
    dict_ids_embeddings = pickle.load(open("data/dict_ids_embeddings.pickle", "rb"))
    VDB_IM = VectorDB_IM()
    logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    start = time.time()
    st.session_state["n_clusters"] = 10
    VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
    stop = time.time()
    print(f"Kmeans indexed in {str(stop - start)}")
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
    query_text = st.text_input(label="**Votre requête:**")
    if hasattr(VDB_IM, 'query_with_kmeans'):
        st.checkbox("**Utiliser l'index inversé**", key="use_ivf")
        if st.session_state["use_ivf"]:
            st.number_input("**Nombre de sondes:**", key="n_probes", value=1, step=1)
    search = st.button("Rechercher")
    if st.session_state["use_ivf"]:
        st.number_input("**Nombre de clusters:**", key="n_clusters", value=10, step=1)
        reindex = st.button("Ré-indexer")
        if reindex:
            VDB_IM.inverted_index = {}
            VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
            st.write(len(VDB_IM.inverted_index))

if search:
    perform_query(VDB_IM, FCLIP, query_text, use_kmeans_query=st.session_state["use_ivf"])
    st.write(st.session_state["log_dataframe"])
