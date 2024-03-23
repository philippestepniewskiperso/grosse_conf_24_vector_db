import streamlit as st
from PIL import Image

from gc_db.vector_db.vector_db_in_memory import VectorDB_IM


def display_result_gallery(images_list: list[str], similarities: list[float], is_bool_list: list[bool],
                           nb_cols: int = 10):
    nb_images = len(images_list)
    nb_rows = int(nb_images / nb_cols)
    image_grid = prepare_grid(nb_rows, nb_cols)
    col = 0
    row = 0
    for i, image_path in enumerate(images_list):
        if ((i % nb_cols) == 0) and (i > 0):
            row += 1
            col = 0
        try:
            with image_grid[row][col]:
                with st.container(border=True):
                    st.image(image_path)
                    is_knn = is_bool_list[i]
                    sim_to_disp = str(round(similarities[i], 2))
                    if hasattr(VectorDB_IM, "query_with_kmeans") or st.session_state["hnsw"]:
                        st.write(
                            f"Similarité: {sim_to_disp} \n **KNN: :{'green' if is_knn else 'red'}[{is_knn}]**")
                    else:
                        st.write(
                            f"Similarité: {sim_to_disp}")
        except IndexError:
            pass
        col += 1


def prepare_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows, gap="large")
    return grid


def image_as_query(uploaded_file):
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, width=200)
    cloth = st.session_state["SEG"].extract_mask_from_image(image)
    with col2:
        st.image(cloth, width=200)
    return cloth
