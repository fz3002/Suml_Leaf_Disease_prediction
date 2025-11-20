from src.data_preparation import Dataset
from src.data_preparation import ImageDataHandler
from torchvision import transforms
import os


def prepare_dataset():
    data_path = os.path.join(os.getcwd(), 'data')
    # download_dataset(data_path)
    dataset: Dataset = Dataset(data_path=data_path)

    dataset.train_test_split()
    train_data: list[tuple[str, str]] = dataset.get_training_data()
    test_data: list[tuple[str, str]] = dataset.get_test_data()
    print('Training data length:', len(train_data))
    print('Test data length:', len(test_data))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_handler: ImageDataHandler = ImageDataHandler(train_data, transform)
    for image, label in train_handler:
        print(image.shape, label)
        break


def main():
    prepare_dataset()


if __name__ == '__main__':
    main()

