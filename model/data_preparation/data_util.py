import kagglehub
import os
import shutil



def download_dataset(dest_path: str) -> None:
    path = kagglehub.dataset_download("warcoder/mango-leaf-disease-dataset")
    print("Path to dataset files:", path)
    if dest_path is None:
        dest_path = os.path.join(os.getcwd(), 'data')
    shutil.copytree(path, dest_path)

