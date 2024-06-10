from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_data
from scripts.split_data import split_data
from scripts.neural_network_cnn import train_model
from scripts.eval_trained_cnn import test_model

folder1 = './data/HAM10000/HAM10000_images_part_1'
folder2 = './data/HAM10000/HAM10000_images_part_2'
destination_folder = './data/HAM10000/HAM10000_images'
metadata_file = './data/HAM10000/HAM10000_metadata.csv'
augmented_folder = './data/HAM10000/HAM10000_images_augmented'

if __name__ == "__main__":
    images, metadata = load_data(folder1, folder2, destination_folder, metadata_file)
    print('loaded')
    images, metadata = preprocess_data(images, metadata, augmented_folder)
    train_X, test_X, train_Y, test_Y = split_data(images, metadata, validation_split=0.2)
    train_images, train_metadata = train_X
    print('train')
    model = train_model(train_images, train_metadata, train_Y)
    print('trained')
    test_images, test_metadata = test_X
    test_model(model, test_images, test_metadata, test_Y)

