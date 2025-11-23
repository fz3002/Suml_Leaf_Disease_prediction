import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from data_preparation.preprocessing.data_util import load_config_file


def remove_duplicates(folder_path, threshold=0):
    folder_path = Path(folder_path)
    hashes = {}

    for img_path in folder_path.glob("*"):
        try:
            hash_val = imagehash.phash(Image.open(img_path))
        except Exception:
            print(f"[remove_duplicates] Nie można wczytać {img_path}, pomijam.")
            continue

        found_duplicate = False

        for existing_path, existing_hash in hashes.items():
            if abs(hash_val - existing_hash) <= threshold:
                print(f"[remove_duplicates] Duplikat: {img_path} → usuwam")
                os.remove(img_path)
                found_duplicate = True
                break

        if not found_duplicate:
            hashes[img_path] = hash_val

def validate_images(folder_path, min_resolution=(100, 100), remove_corrupt=True):
    folder_path = Path(folder_path)

    for img_path in folder_path.glob("*"):
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)
        except Exception:
            if remove_corrupt:
                print(f"[validate_images] Uszkodzony plik: {img_path} → usunięty")
                os.remove(img_path)
            continue

        if img.size[0] < min_resolution[0] or img.size[1] < min_resolution[1]:
            print(f"[validate_images] Za mała rozdzielczość: {img_path} → usunięty")
            os.remove(img_path)


def white_balance(img):
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)

    corrected = wb.balanceWhite(img)
    return corrected


def gamma_correction(img, gamma=1.0):
    if gamma <= 0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img, method: str="gaussian", kernel_size: int=5):
    if method == "gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(img, kernel_size)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Nieznana metoda denoise: {method}")

def preprocess_directory(dir_path: str, preprocess: dict) -> None:
    remove_duplicates(dir_path)
    validate_images(dir_path)

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        if not os.path.isfile(file_path):
            continue
        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARN] Nie można wczytać pliku: {file_path}")
            continue

        if preprocess['denoise']['enable']:
            method = preprocess['denoise']['method']
            kernel_size = preprocess['denoise']['kernel_size']
            img = denoise(img, method=method, kernel_size=kernel_size)

        if preprocess['clahe']['enable']:
            clip_limit = preprocess['clahe']['clip_limit']
            tile_grid_size = preprocess['clahe']['tile_grid_size']
            img = apply_clahe(img, clip_limit=clip_limit,
                                   tile_grid_size=tuple(tile_grid_size))

        if preprocess['gamma_correction']['enable']:
            gamma = preprocess['gamma_correction']['gamma']
            img = gamma_correction(img, gamma=gamma)

        if preprocess['white_balance']['enable']:
            img = white_balance(img)

        cv2.imwrite(file_path, img)

    print(f"[INFO] Preprocessing zakończony dla folderu: {dir_path}")


def preprocess_images() -> None:
    config = load_config_file()

    train_path = config['path']['3_processed']['train']
    val_path = config['path']['3_processed']['val']
    test_path = config['path']['3_processed']['test']

    for class_name in os.listdir(train_path):
        full_path = os.path.join(train_path, class_name)
        preprocess_directory(full_path, config['preprocess'])

    for class_name in os.listdir(val_path):
        full_path = os.path.join(val_path, class_name)
        preprocess_directory(full_path, config['preprocess'])

    for class_name in os.listdir(test_path):
        full_path = os.path.join(test_path, class_name)
        preprocess_directory(full_path, config['preprocess'])

