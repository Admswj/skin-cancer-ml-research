import numpy as np
from scripts.preprocess_data import lesion_type_dict


def split_X(images, metadata, validation_split=0.2):
    data_size = len(images)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    val_size = int(data_size * validation_split)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_images = images[train_indices]
    val_images = images[val_indices]

    train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
    val_metadata = metadata.iloc[val_indices].reset_index(drop=True)

    return (train_images, train_metadata), (val_images, val_metadata)


def extract_labels(metadata):
    labels = metadata['dx'].map({v['code']: k for k, v in lesion_type_dict.items()})
    return labels


def split_data(images, metadata, validation_split=0.2):
    (train_images, train_metadata), (val_images, val_metadata) = split_X(images, metadata, validation_split)

    train_labels = extract_labels(train_metadata)
    val_labels = extract_labels(val_metadata)

    train_metadata = train_metadata.drop(columns=['dx'])
    val_metadata = val_metadata.drop(columns=['dx'])

    return (train_images, train_metadata), (val_images, val_metadata), train_labels, val_labels
