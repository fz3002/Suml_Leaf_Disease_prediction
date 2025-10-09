import kagglehub
import os
from PIL import Image
import matplotlib.pyplot as plt


path = kagglehub.dataset_download("warcoder/mango-leaf-disease-dataset")

print("Path to dataset files:", path)

image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
for img_file in image_files[:5]:  # Display first 5 images
    img_path = os.path.join(path, img_file)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(img_file)
    plt.show()
