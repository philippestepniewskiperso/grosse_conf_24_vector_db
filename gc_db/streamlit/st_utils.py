import os.path
from datetime import datetime
import time
from typing import Any

import PIL
from PIL.Image import Image
from deep_translator import GoogleTranslator
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import streamlit as st
import gc_db.streamlit.st_creators as stc
from gc_db.utils.utils import get_path_from_image_id, check_if_is_ann
from gc_db.config import ROOT_DIR
import pandas as pd

from gc_db.vector_db.vector_db_in_memory import VectorDB_IM


def load_logs_of_the_day_if_exists():
    current_date = get_slugified_current_date()
    log_file = os.path.join(ROOT_DIR, "logs", current_date)
    if os.path.isfile(log_file):
        st.session_state["log_dataframe"] = pd.read_parquet(log_file)


def get_slugified_current_date():
    aujourdhui = datetime.now()
    return aujourdhui.strftime("%Y-%m-%d")


def dump_logs_to_file(log_df: pd.DataFrame):
    file_name = get_slugified_current_date()
    log_df.to_parquet(os.path.join(ROOT_DIR, "logs", file_name))


def log_query(query: str, query_time: float, recall: float):
    if isinstance(query, Image):
        query = "image_query"
    now = datetime.now()
    dt_index = pd.to_datetime(now)
    use_ivf = st.session_state["use_ivf"] if "use_ivf" in st.session_state else False
    n_clusters = st.session_state["n_clusters"] if use_ivf else None
    n_probes = st.session_state["n_probes"] if use_ivf else None
    log_row = {"Requête": query, "Index inversé": [str(use_ivf)], "K": [st.session_state["k"]],
               "nombre de sondes": [n_probes], "n_clusters": [n_clusters],
               "temps de requête": [query_time], "rappel": [recall]}
    if "log_dataframe" in st.session_state:
        log_df = st.session_state["log_dataframe"].copy()
        log_df = pd.concat([log_df, pd.DataFrame(log_row, index=[dt_index])])
        st.session_state["log_dataframe"] = log_df
    else:
        log_df = pd.DataFrame(log_row, index=[dt_index])
        st.session_state["log_dataframe"] = log_df
    dump_logs_to_file(log_df)


def perform_query(vdb: VectorDB_IM, fclip: FashionCLIP, query: str | PIL.Image.Image, use_kmeans_query: bool):
    k = st.session_state["k"]
    if isinstance(query, PIL.Image.Image):
        embeded_query = fclip.encode_images([query], 1)[0]
    else:
        embeded_query = fclip.encode_text([query], 1)[0]
    start = time.time()
    if use_kmeans_query:
        try:
            nn = vdb.query_with_kmeans(embeded_query, n_probes=st.session_state["n_probes"], k=k)
        except TypeError:
            nn = vdb.query_with_kmeans(embeded_query, k=k)
    else:
        nn = vdb.query(embeded_query, k=k)
    end = time.time()

    lasted = np.round(end - start, 5)
    ids, similarities = zip(*nn)
    image_pathes = [get_path_from_image_id(dist_id) for dist_id in ids]
    is_knn, recall = get_true_nn_for_query(st.session_state["VDB_IM"], nn, embeded_query, k=k)
    st.info(f"Temps d'éxecution de la requête : **{lasted} seconds**")
    stc.display_result_gallery(image_pathes, similarities, is_knn, nb_cols=5)
    log_query(query, lasted, recall)


def get_true_nn_for_query(vdb: VectorDB_IM, returned_nn_ids: list[int], embeded_query: np.array, k: int = 20) -> tuple[
    list[bool], float | Any]:
    true_nn = vdb.query(embeded_query, k=k)
    is_knn = [check_if_is_ann(nn_id[0], true_nn) for nn_id in returned_nn_ids]
    recall = np.array(is_knn).sum() / k
    return is_knn, recall


def translate_query(query: str) -> str:
    return GoogleTranslator(source='fr', target='en').translate(query)
