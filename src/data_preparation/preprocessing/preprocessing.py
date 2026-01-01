import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import imagehash
from pathlib import Path
from src.data_preparation.preprocessing.data_util import load_config_file


def remove_duplicates(folder_path: str, threshold: int = 0) -> None:
    """
    Remove duplicate images from a folder using perceptual hashing.

    The function computes a perceptual hash (pHash) for each image in the given
    directory and compares it against hashes of previously processed images.
    If the Hamming distance between two hashes is less than or equal to
    `threshold`, the image is considered a duplicate and removed.

    :param folder_path: Path to the directory containing images to be checked for duplicates.
    :param threshold: Maximum allowed Hamming distance between two perceptual hashes for images
        to be considered duplicates. A value of 0 means only exact pHash matches
        are treated as duplicates.
    :return: None
    """

    folder_path: Path = Path(folder_path)
    hashes: dict[Path, imagehash.ImageHash] = {}

    # Iterate over all files in the directory
    for img_path in folder_path.glob("*"):
        try:
            # Compute perceptual hash of the image
            hash_val: imagehash.ImageHash = imagehash.phash(Image.open(img_path))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[WARNING] remove_duplicates() - can't open {img_path} : {e}")
            continue

        found_duplicate: bool = False

        # Compare with previously computed hashes
        for existing_path, existing_hash in hashes.items():
            if abs(hash_val - existing_hash) <= threshold:
                print(f"[INFO] remove_duplicates() - removing duplicate: {img_path}")
                os.remove(img_path)
                found_duplicate = True
                break

        # Store hash if the image is unique
        if not found_duplicate:
            hashes[img_path] = hash_val


def validate_images(folder_path: str, min_resolution: tuple[int, int] = (100, 100), remove_corrupt: bool = True) -> None:
    """
    Validate images in a directory and optionally remove invalid files.

    The function checks each image in the specified directory for:
    1. File integrity (whether the image can be opened and verified).
    2. Minimum resolution requirements.

    Images that fail validation are optionally removed from disk.

    :param folder_path: Path to the directory containing images to be validated.
    :param min_resolution: Minimum allowed image resolution in the form (width, height).
    :param remove_corrupt: If True, corrupted or unreadable images are deleted.
    :return: None
    """

    folder_path: Path = Path(folder_path)

    # Iterate over all files in the directory
    for img_path in folder_path.glob("*"):
        try:
            # Verify image integrity without decoding pixel data
            img = Image.open(img_path)
            img.verify()

            # Reopen the image after verification for further checks
            img = Image.open(img_path)
        except Exception:
            if remove_corrupt:
                print(f"[validate_images] Corrupted file detected: {img_path} -> removed")
                os.remove(img_path)
            continue

        # Check minimum resolution constraints
        if img.size[0] < min_resolution[0] or img.size[1] < min_resolution[1]:
            print(f"[validate_images] Image resolution too small: {img_path} -> removed")
            os.remove(img_path)


def white_balance(img: np.ndarray) -> np.ndarray:
    """
    Apply gray-world white balance correction to an image.

    This function uses OpenCV's Gray World White Balance algorithm to
    normalize color channels under the assumption that the average
    scene color should be neutral gray.

    :param img: Input image as a NumPy array in BGR color space
    :return: White-balanced image in BGR color space.
    """
    # Create Gray World White Balance object
    wb = cv2.xphoto.createGrayworldWB()

    # Ignore highly saturated pixels during white balance estimation
    wb.setSaturationThreshold(0.99)

    # Apply white balance correction
    corrected: np.ndarray = wb.balanceWhite(img)

    return corrected


def gamma_correction(img, gamma=1.0):
    """
    Gamma correction pojaśnia albo pociemnia obraz w sposób nieliniowy
    :param img: Obraz
    :param gamma: Współczynnik jasności
    :return: Zmieniony obraz
    """
    if gamma <= 0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Funkcja wyrównuje histogram z ograniczeniem kontrastu
    :param img: Obraz
    :param clip_limit:
    :param tile_grid_size:
    :return:
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # Konwersja obrazu dp LAB
    luminosity, green_red, blue_yellow = cv2.split(lab) # Bierzemy każdą wartość
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    luminosity_changed = clahe.apply(luminosity) # Aplikujemy na kanał jasności
    merged = cv2.merge((luminosity_changed, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img, method: str="gaussian", kernel_size: int=5):
    """

    :param img:
    :param method:
    :param kernel_size:
    :return:
    """

    if method == "gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(img, kernel_size)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Nieznana metoda denoise: {method}")


def preprocess_directory(dir_path: str, preprocess: dict) -> None:
    """

    :param dir_path:
    :param preprocess:
    :return:
    """

    remove_duplicates(dir_path)
    validate_images(dir_path)

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        if not os.path.isfile(file_path):
            continue
        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARNING] Nie można wczytać pliku: {file_path}")
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
    """

    :return:
    """

    config = load_config_file()

    project_path: str = config['path']["project_path"]

    train_path: str = os.path.join(project_path, config['path']['processed']['train'])
    val_path: str = os.path.join(project_path, config['path']['processed']['val'])
    test_path: str = os.path.join(project_path, config['path']['processed']['test'])

    for class_name in os.listdir(train_path):
        full_path: str = os.path.join(train_path, class_name)
        preprocess_directory(full_path, config['preprocess'])

    for class_name in os.listdir(val_path):
        full_path: str = os.path.join(val_path, class_name)
        preprocess_directory(full_path, config['preprocess'])

    for class_name in os.listdir(test_path):
        full_path: str = os.path.join(test_path, class_name)
        preprocess_directory(full_path, config['preprocess'])
