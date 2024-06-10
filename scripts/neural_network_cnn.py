import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ReLU, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Progbar
import gc

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, metadata, labels, batch_size, shuffle=True):
        self.images = images
        self.metadata = metadata
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = self.images[batch_indices]
        batch_metadata = self.metadata[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_metadata = transform_metadata(batch_metadata)
        combined_input = combine_inputs(batch_images, batch_metadata)
        return combined_input, batch_labels

    def on_epoch_end(self):
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_model(image_shape, metadata_shape, num_classes, M=2, N=2, K=2, learning_rate=0.001):
    combined_input = Input(shape=(np.prod(image_shape) + metadata_shape,), name='combined_input')
    image_input = Lambda(lambda x: x[:, :np.prod(image_shape)])(combined_input)
    metadata_input = Lambda(lambda x: x[:, np.prod(image_shape):])(combined_input)
    image_input = Lambda(lambda x: tf.reshape(x, [-1] + list(image_shape)))(image_input)

    x = image_input
    for _ in range(M):
        for _ in range(N):
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    for _ in range(K):
        x = Dense(128)(x)
        x = ReLU()(x)

    y = metadata_input
    for _ in range(K):
        y = Dense(128)(y)
        y = ReLU()(y)

    combined = concatenate([x, y])
    z = Dense(128)(combined)
    z = ReLU()(z)
    z = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[combined_input], outputs=z)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def transform_metadata(metadata):
    categorical_features = ['dx_type', 'sex', 'localization']
    numeric_features = ['age']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ])
    metadata_transformed = preprocessor.fit_transform(metadata)
    return metadata_transformed

def combine_inputs(images, metadata):
    flattened_images = images.reshape((images.shape[0], -1))
    combined_input = np.concatenate([flattened_images, metadata], axis=1)
    return combined_input

def train_model(images, metadata, labels, validation_split=0.2, batch_size=32):
    metadata_transformed = transform_metadata(metadata)
    combined_input = combine_inputs(images, metadata_transformed)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    param_grid = {
        'M': [2, 3],
        'N': [2, 3],
        'K': [2, 3],
        'batch_size': [16, 32],
        'epochs': [50, 100],
        'learning_rate': [1e-3, 1e-4]
    }

    def create_model(M=2, N=2, K=2, learning_rate=0.001):
        image_shape = images.shape[1:]
        metadata_shape = metadata_transformed.shape[1]
        num_classes = len(np.unique(labels))
        return build_model(image_shape, metadata_shape, num_classes, M, N, K, learning_rate)

    model = KerasClassifier(build_fn=create_model, verbose=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=skf, scoring='accuracy')

    best_model = None
    best_score = 0
    best_params = None

    for train_index, test_index in skf.split(combined_input, labels):
        train_combined, val_combined = combined_input[train_index], combined_input[test_index]
        train_labels, val_labels = labels[train_index], labels[test_index]

        train_generator = DataGenerator(images[train_index], metadata[train_index], train_labels, batch_size)
        val_generator = DataGenerator(images[test_index], metadata[test_index], val_labels, batch_size, shuffle=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        callbacks = [early_stopping, reduce_lr]

        print("Starting Grid Search on current fold...")
        grid_result = grid.fit(
            train_combined, train_labels,
            class_weight=class_weights,
            callbacks=callbacks,
            validation_data=(val_combined, val_labels)
        )

        print("Finished Grid Search on current fold.")
        print(f"Fold Score: {grid_result.best_score_} using {grid_result.best_params_}")

        if grid_result.best_score_ > best_score:
            best_score = grid_result.best_score_
            best_params = grid_result.best_params_
            best_model = grid_result.best_estimator_.model

        del train_combined, val_combined, train_labels, val_labels
        gc.collect()

    print(f"Best overall score: {best_score} using {best_params}")

    batch_size = best_params['batch_size']
    epochs = best_params['epochs']
    learning_rate = best_params['learning_rate']

    train_size = int((1 - validation_split) * len(labels))
    train_generator = DataGenerator(images[:train_size], metadata[:train_size], labels[:train_size], batch_size)
    val_generator = DataGenerator(images[train_size:], metadata[train_size:], labels[train_size:], batch_size, shuffle=False)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    callbacks = [early_stopping, reduce_lr]

    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)

    print("Starting final training...")
    progbar = Progbar(target=epochs)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        best_model.fit(
            train_generator,
            epochs=1,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        progbar.update(epoch + 1)

    print(f"Best parameters after final training: {best_params}")

    return best_model

# Example usage
# metadata = pd.read_csv('path_to_metadata.csv')
# images, labels = load_images_and_labels() # Implement this function to load your images and labels
# model = train_model(images, metadata, labels)
