import logging as logger
import pickle
import time
from PIL import Image

import streamlit as st
from fashion_clip.fashion_clip import FashionCLIP

from gc_db.streamlit.st_utils import perform_query, translate_query
from gc_db.vector_db.vector_db_in_memory import VectorDB_IM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.session_state["k"] = 20

if "is_initiated" not in st.session_state:
    logger.info("INITIATING")
    dict_ids_embeddings = pickle.load(open("data/dict_ids_embeddings_full.pickle", "rb"))
    VDB_IM = VectorDB_IM()
    logger.info("Loading vector to memory db : " + str(len(dict_ids_embeddings.keys())))
    _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
    if hasattr(VDB_IM, "init_kmeans_index"):
        st.session_state["n_clusters"] = 10
        start = time.time()
        VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
        stop = time.time()
        logger.info(f"Kmeans indexed in {str(stop - start)}")
    st.session_state["VDB_IM"] = VDB_IM
    st.session_state["FCLIP"] = FashionCLIP('fashion-clip')
    st.session_state["is_initiated"] = True
else:
    VDB_IM = st.session_state["VDB_IM"]
    FCLIP = st.session_state["FCLIP"]

st.set_page_config(layout="wide")
st.title("MOTEUR DE RECHERCHE MULTI-MODAL")
with st.sidebar:
    st.image("./data/assets/gc_logo.webp")
    query_text = st.text_input(label="**Votre requête:**")
    if hasattr(VDB_IM, 'query_with_kmeans'):
        st.checkbox("**Utiliser l'index inversé**", key="use_ivf")
        if st.session_state["use_ivf"]:
            st.number_input("**Nombre de sondes:**", key="n_probes", value=1, step=1)

        if st.session_state["use_ivf"]:
            st.number_input("**Nombre de clusters:**", key="n_clusters", value=10, step=1)
            reindex = st.button("Ré-indexer")
            if reindex:
                VDB_IM.inverted_index = {}
                VDB_IM.init_kmeans_index(nb_clusters=st.session_state["n_clusters"])
                st.write(len(VDB_IM.inverted_index))
    search = st.button("Rechercher")

tab1, tab2 = st.tabs(["Texte", "Image"])
use_ivf = st.session_state["use_ivf"] if "use_ivf" in st.session_state else False
with tab1:
    if search:
        logger.info(f"Requête {query_text} ")
        translated_query_text = translate_query(query_text)
        logger.info(f"Translated {query_text} to {translated_query_text}")
        perform_query(VDB_IM, FCLIP, translated_query_text, use_kmeans_query=use_ivf)
        st.write(st.session_state["log_dataframe"])

with tab2:
    uploaded_file = st.file_uploader("Choose an image to upload", accept_multiple_files=False)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        perform_query(VDB_IM, FCLIP, image, use_kmeans_query=use_ivf)
