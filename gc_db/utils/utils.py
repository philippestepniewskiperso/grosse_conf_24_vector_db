import os


def get_path_from_image_id(id: int, images_folder: str = "data/images"):
    id = str(id).zfill(10)
    subfolder = id[0:3]
    path = os.path.join(images_folder, subfolder, id + ".jpg")
    return path


def check_if_is_ann(ann_id: int, knn_list_ids: list[int]) -> bool:
    ids, distances = zip(*knn_list_ids)
    print(ann_id, ids)
    return ann_id in ids
