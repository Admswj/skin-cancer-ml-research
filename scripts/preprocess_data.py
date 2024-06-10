import os
import shutil

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

lesion_type_dict = {
    1: {'code': 'nv', 'name': 'Melanocytic nevi'},
    2: {'code': 'mel', 'name': 'Melanoma'},
    3: {'code': 'bkl', 'name': 'Benign keratosis-like lesions'},
    4: {'code': 'bcc', 'name': 'Basal cell carcinoma'},
    5: {'code': 'akiec', 'name': 'Actinic keratoses'},
    6: {'code': 'vasc', 'name': 'Vascular lesions'},
    7: {'code': 'df', 'name': 'Dermatofibroma'}
}


def augment_images(images, metadata, augmented_folder, batch_size=2):
    """
    Augment images by applying various transformations and save them to a folder.

    Parameters:
    - images: NumPy array of images to be augmented.
    - metadata: DataFrame containing metadata of the images.
    - batch_size: Number of augmentations to be generated per image.

    Returns:
    - all_images: NumPy array containing original and augmented images.
    - all_metadata: DataFrame containing metadata for original and augmented images.
    """

    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: np.rot90(x, k=np.random.choice([1, 2, 3])),
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=(1, 1.2),
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create the folder for augmented images if it does not exist
    if os.path.exists(augmented_folder):
        shutil.rmtree(augmented_folder)
    os.makedirs(augmented_folder)

    # DataFrames and lists to store augmented images and metadata
    augmented_metadata = pd.DataFrame()
    augmented_images = []

    # Augment each image
    for i in range(len(images)):
        image = images[i]
        image = img_to_array(image).reshape((1,) + image.shape)

        # Generate augmented images
        aug_iter = datagen.flow(image, batch_size=batch_size)
        aug_images = [next(aug_iter)[0] for _ in range(batch_size)]

        # print(aug_images)

        for j, aug_image in enumerate(aug_images):
            aug_image = aug_image.astype(np.uint8)
            img = array_to_img(aug_image)

            # Save augmented image
            aug_image_id = f"{metadata.iloc[i]['image_id']}_aug_{j}"
            img_path = os.path.join(augmented_folder, f"{aug_image_id}.jpg")
            img.save(img_path)

            # Append metadata for augmented image
            new_metadata = pd.DataFrame({
                'image_id': [aug_image_id],
                'lesion_id': [metadata.iloc[i]['lesion_id']],
                'dx': [metadata.iloc[i]['dx']],
                'dx_type': [metadata.iloc[i]['dx_type']],
                'age': [metadata.iloc[i]['age']],
                'sex': [metadata.iloc[i]['sex']],
                'localization': [metadata.iloc[i]['localization']]
            })
            augmented_metadata = pd.concat([augmented_metadata, new_metadata], ignore_index=True)

            augmented_images.append(aug_image)

    # Combine original and augmented images
    all_images = np.concatenate((images, np.array(augmented_images)), axis=0)
    all_metadata = pd.concat([metadata, augmented_metadata], ignore_index=True)

    return all_images, all_metadata


def sanitize_data(metadata):
    """
    Clean metadata by filling missing values and drop redundant columns.

    Parameters:
    - metadata: DataFrame containing metadata of the images.
    """
    # Drop redundant columns from learning standpoint
    metadata = metadata.drop(columns=['lesion_id', 'image_id'], errors='ignore')

    # Fill missing age with mean age
    metadata['age'].fillna(metadata['age'].mean(), inplace=True)

    # Fill missing values for categorical columns by drawing from the estimated distribution
    categorical_columns = ['dx_type', 'sex', 'localization']

    for col in categorical_columns:
        # Calculate the value counts and normalize to get probabilities
        value_counts = metadata[col].value_counts(normalize=True)

        # Generate random choices based on the probability distribution
        missing_indices = metadata[metadata[col].isna()].index
        random_choices = np.random.choice(value_counts.index, size=len(missing_indices), p=value_counts.values)

        # Fill missing values with the random choices
        metadata.loc[missing_indices, col] = random_choices

    return metadata


def preprocess_data(images, metadata, augmented_folder):
    """
    Arranges preprocess steps into a pipeline.

    Parameters:
    - images: NumPy array of images to be preprocessed.
    - metadata: DataFrame containing metadata of the images.

    Returns:
    - images: NumPy array of preprocessed images.
    - metadata: DataFrame containing cleaned metadata.
    """
    print('preprocessing')
    images, metadata = augment_images(images, metadata, augmented_folder)
    print('augmented')
    sanitize_data(metadata)
    return images, metadata
