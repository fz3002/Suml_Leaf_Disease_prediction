from data_preparation.preprocessing.data_util import *
from data_preparation.preprocessing.preprocessing import *


def download_dataset_to_local_dir() -> None:
    config = load_config_file()
    raw_path = config.get('path').get('raw')
    download_dataset(dest_path=raw_path)


def main():
    # download_dataset_to_local_dir()
    # prepare_raw_dataset()
    # preprocess_images()
    pass


if __name__ == '__main__':
    main()
