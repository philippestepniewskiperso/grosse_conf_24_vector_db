import streamlit as st
from gc_db.utils.flex_logging import stream_handler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

def display_result_gallery(images_list: list[str], nb_cols: int = 10):
    nb_images = len(images_list)
    nb_rows = int(nb_images / nb_cols)
    image_grid = prepare_grid(nb_cols, nb_rows)
    col = 0
    row = 0
    for i, image_path in enumerate(images_list):
        if ((i % nb_cols) == 0) and (i > 0):
            row += 1
            col = 0
        try:
            image_grid[row][col].image(image_path)
        except IndexError:
            pass
        col += 1


def prepare_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

def prepare_check_boxes():
    st.session_state["use_kmeans_index"] = st.checkbox("Utiliser un index K-means")
    if st.session_state["use_kmeans_index"]:
        st.session_state["n_clusters"] = st.number_input("Nombre de clusters")
        st.session_state["add_nprobes"] = st.checkbox("Paramétrage N-sondes")
        if st.session_state["add_nprobes"]:
            st.session_state["n_probes"] = st.number_input("Nombre de sondes")
