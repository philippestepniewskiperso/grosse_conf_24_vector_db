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


def set_style_css(major_font_size_px: int = 30, minor_font_size_px: int = 25):
    tabs_font_css = """
    <style>
    div[class*="stTextArea"] label p{
      font-size: {major_font_size_px}px;
    }

    div[class*="stTextInput"] label p{
      font-size: {major_font_size_px}px;
    }

    div[class*="stNumberInput"] label p{
      font-size: {major_font_size_px}px;
    }

    div[class*="stButton"] label p{
      font-size: {major_font_size_px}px;
    }

    div[class*=".st-emotion-cache-16idsys"] label p{
      font-size: {major_font_size_px}px;
    }

    div[class*=".st-emotion-cache-1vbkxwb"] label p{
      font-size: {major_font_size_px}px;
    }

    .st-emotion-cache-1vbkxwb strong {
        font-size: {major_font_size_px}px; /* Remplacez la taille par celle que vous désirez */
    }

    .st-ag input {
        font-size: {major_font_size_px}px; /* Remplacez la taille par celle que vous désirez */
    }

    .st-emotion-cache-5rimss p,
    .st-emotion-cache-5rimss strong {
        font-size: {minor_font_size_px}px; /* Remplacez la taille par celle que vous désirez */
    }

    .st-emotion-cache-16idsys p {
        font-size: {minor_font_size_px}px; /* Remplacez la taille par celle que vous désirez */
    }
    </style>
    """
    tabs_font_css = tabs_font_css.replace("{major_font_size_px}", str(major_font_size_px))
    tabs_font_css = tabs_font_css.replace("{minor_font_size_px}", str(minor_font_size_px))

    st.write(tabs_font_css, unsafe_allow_html=True)
