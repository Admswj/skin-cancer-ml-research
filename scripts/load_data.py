import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np


def load_data(folder1, folder2, destination_folder, metadata_file):
    """
    Load data from two folders of images and a metadata CSV file.

    Parameters:
    - folder1: Path to the first folder of images.
    - folder2: Path to the second folder of images.
    - destination_folder: Path to the destination folder where merged images will be stored.
    - metadata_file: Path to the metadata CSV file containing image labels.

    Returns:
    - images: NumPy array of resized images.
    - labels: NumPy array of corresponding labels.
    - metadata: DataFrame containing metadata excluding the label column.
    """

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)

    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            source = os.path.join(folder, filename)
            destination = os.path.join(destination_folder, filename)
            shutil.copy(source, destination)

    metadata = pd.read_csv(metadata_file)
    metadata = metadata.drop_duplicates()

    images = []
    for index, row in metadata.iterrows():
        img_path = os.path.join(destination_folder, row['image_id'] + '.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((256, 256))
            images.append(np.array(image))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    images = np.array(images)

    return images, metadata
