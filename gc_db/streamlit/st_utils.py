from datetime import datetime, time
from typing import Any
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import streamlit as st
import gc_db.streamlit.st_creators as stc
from gc_db.utils.utils import get_path_from_image_id, check_if_is_ann

import pandas as pd

from gc_db.vector_db.vector_db import VectorDB


def log_query(query_time: float, recall: float):
    now = datetime.now()
    dt_index = pd.to_datetime(now)
    n_clusters = st.session_state["n_clusters"] if st.session_state["use_ivf"] else None
    n_probes = st.session_state["n_probes"] if st.session_state["use_ivf"] else None
    log_row = {"Index inversé": [str(st.session_state["use_ivf"])], "K": [st.session_state["k"]],
               "nombre de sondes": [n_probes], "n_clusters": [n_clusters],
               "temps de requête": [query_time], "rappel": [recall]}
    if "log_dataframe" in st.session_state:
        old_df = st.session_state["log_dataframe"].copy()
        old_df = pd.concat([old_df, pd.DataFrame(log_row, index=[dt_index])])
        st.session_state["log_dataframe"] = old_df
    else:
        st.session_state["log_dataframe"] = pd.DataFrame(log_row, index=[dt_index])


def perform_query(vdb: VectorDB, fclip: FashionCLIP, query_text: str, use_kmeans_query: bool):
    k = st.session_state["k"]
    embeded_query = fclip.encode_text([query_text], 1)[0]
    start = time.time()
    if use_kmeans_query:
        nn = vdb.query_with_kmeans(embeded_query, n_probes=st.session_state["n_probes"], k=k)
    else:
        nn = vdb.query(embeded_query, k=k)
    end = time.time()

    lasted = np.round(end - start, 3)
    ids, similarities = zip(*nn)
    image_pathes = [get_path_from_image_id(dist_id) for dist_id in ids]
    is_knn, recall = get_true_nn_for_query(vdb, nn, embeded_query, k=k)
    st.info(f"Temps d'éxecution de la requête : **{lasted} seconds**")
    stc.display_result_gallery(image_pathes, similarities, is_knn, nb_cols=5)
    log_query(lasted, recall)


def get_true_nn_for_query(vdb: VectorDB, returned_nn_ids: list[int], embeded_query: np.array, k: int = 20) -> tuple[
    list[bool], float | Any]:
    true_nn = vdb.query(embeded_query, k=k)
    is_knn = [check_if_is_ann(nn_id[0], true_nn) for nn_id in returned_nn_ids]
    recall = np.array(is_knn).sum() / k
    return is_knn, recall
