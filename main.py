from data_preparation.data_download import *
from data_preparation.dataset import Dataset
import os


def prepare_dataset():
    data_path = os.path.join(os.getcwd(), 'data')
    # download_dataset(data_path)
    dataset: Dataset = Dataset(data_path=data_path)
    dataset.visualize(0)
    dataset.show_four_images()
    dataset.show_data_stats()


# Test

def main():
    prepare_dataset()



if __name__ == '__main__':
    main()

