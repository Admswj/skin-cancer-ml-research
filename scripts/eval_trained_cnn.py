import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def test_model(model, test_images, test_metadata, test_labels):
    """
    Evaluate the performance of the trained model on the test data.

    Parameters:
    model (tf.keras.Model): Trained Keras model.
    test_images (np.ndarray): Array of test images.
    test_metadata (np.ndarray): Array of test metadata.
    test_labels (np.ndarray): Array of test labels.

    Returns:
    dict: A dictionary containing accuracy, classification report, and confusion matrix.
    """

    # Predict labels for test data
    predictions = model.predict([test_images, test_metadata])
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)

    # Generate classification report
    class_report = classification_report(test_labels, predicted_labels)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)

    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }