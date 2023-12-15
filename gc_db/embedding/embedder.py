import pickle
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from gc_db.utils.flex_logging import stream_handler
import os
import logging

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class Embedder:
    def __init__(self):
        self.fclip_embedding = FashionCLIP('fashion-clip')

    @staticmethod
    def list_image_pathes(path_to_folder: str) -> [str]:
        images_list = []
        for parent_folder, subfolders, files in os.walk(path_to_folder):
            for file in files:
                full_path = os.path.join(parent_folder, file)
                images_list.append(full_path)
        return images_list

    def extract_embedding(self, image_path_list: [str]):
        image_embeddings = self.fclip_embedding.encode_images(image_path_list, batch_size=32)
        return image_embeddings

    @staticmethod
    def extract_product_id(full_path) -> int:
        file_name_with_extension = os.path.basename(full_path)
        name_without_extension, _ = os.path.splitext(file_name_with_extension)
        return int(name_without_extension)

    def create_product_id_embedding_dict(self, image_path_list: str, n_samples: int = None) -> dict[int:np.array]:
        image_pathes_list = self.list_image_pathes(image_path_list)
        products_ids = [self.extract_product_id(path) for path in image_pathes_list]
        logger.info("product_ids :" + str(products_ids))
        if n_samples is not None:
            dict_ids_embeddings = dict(list(zip(products_ids, self.extract_embedding(image_pathes_list[0:10]))))
        else:
            dict_ids_embeddings = dict(
                list(zip(products_ids[0:n_samples], self.extract_embedding(image_pathes_list[0:n_samples]))))
        return dict_ids_embeddings

    @staticmethod
    def serialize_embeddings(dict_ids_embeddings: dict[int:np.array], save_path: str):
        pickle.dump(dict_ids_embeddings, open(save_path, "wb"))


if __name__ == "__main__":
    logger.info("Starting embedding extraction")
    emb = Embedder()
    dict_ids_embeddings = emb.create_product_id_embedding_dict("images/")
    logger.info(dict_ids_embeddings)
    emb.serialize_embeddings(dict_ids_embeddings, "data/dict_ids_embeddings.pickle")
