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


def validate_images(folder_path: str, min_resolution: tuple[int, int]=(100, 100), remove_corrupt: bool = True) -> None:
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


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Gamma correction adjusts image brightness in a non-linear manner,
    allowing the image to be brightened or darkened depending on the
    gamma value.
        - gamma > 1.0 darkens the image
        - gamma < 1.0 brightens the image
        - gamma <= 0 the input image is returned unchanged.

    :param img: Input image as a numpy array.
    :param gamma: Gamma correction factor.
    :return: Corrected image with the same shape as the input.
    """
    # Invalid gamma values result in no operation
    if gamma <= 0:
        return img

    # Inverse gamma used to build the lookup table
    inv_gamma: float = 1.0 / gamma

    # Build lookup table for pixel value transformation
    table: np.ndarray = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype="uint8")

    # Apply gamma correction using a lookup table
    return cv2.LUT(img, table)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    The function converts the input image from BGR to LAB color space,
    applies CLAHE to the L (luminance) channel only, and then converts
    the image back to BGR. This enhances local contrast while limiting
    noise amplification.

    :param img: Input image as a NumPy array in BGR color space.
    :param clip_limit: Threshold for contrast limiting. Higher values increase contrast but may also amplify noise.
    :param tile_grid_size : Size of the grid for histogram equalization (in tiles). Smaller tiles increase local contrast.
    :return: Image with CLAHE applied, in BGR color space.
    """
    # Convert image from BGR to LAB color space
    lab: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels: L (luminance), A (green-red), B (blue-yellow)
    luminosity: np.ndarray
    green_red: np.ndarray
    blue_yellow: np.ndarray
    luminosity, green_red, blue_yellow = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the luminance channel
    luminosity_changed: np.ndarray = clahe.apply(luminosity)

    # Merge channels back into LAB image
    merged: np.ndarray = cv2.merge((luminosity_changed, green_red, blue_yellow))

    # Convert image back to BGR color space
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
    """
    Apply noise reduction to an image using the specified denoising method.

    :param img: Input image as a numpy array.
    :param method: Denoising method to apply. Supported values: gaussian, median, bilateral
    :param kernel_size: Size of the kernel used for Gaussian and median filtering.
    :return: Denoised image with the same shape as the input.
    """
    # Apply Gaussian blur
    if method == "gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Apply median filtering
    elif method == "median":
        return cv2.medianBlur(img, kernel_size)

    # Apply bilateral filtering (edge-preserving)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsupported denoising method
    else:
        raise ValueError(f"Unknown denoising method: {method}")


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
