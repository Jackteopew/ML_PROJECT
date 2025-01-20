# -*- coding: utf-8 -*-
from keras.models import Sequential
# initialize nn
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
# convert pooling features space to large feature vector for fully
# connected layer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from collections import Counter


# basic cnn
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(25, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=None,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')
# print(test_datagen);
labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

labels2 = (test_set.class_indices)
print(labels2)

history = model.fit(training_set,
                    steps_per_epoch=len(training_set),
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=len(test_set))


def plot_training_curves(history):
    """Plot training vs validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_dataset_distribution(training_set, test_set):
    """Plot distribution of images across classes in training and test sets"""
    class_names = list(training_set.class_indices.keys())

    # Count samples per class
    train_counts = Counter(training_set.classes)
    test_counts = Counter(test_set.classes)

    # Prepare data for plotting
    train_samples = [train_counts[i] for i in range(len(class_names))]
    test_samples = [test_counts[i] for i in range(len(class_names))]

    # Create bar plot
    plt.figure(figsize=(15, 6))
    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width / 2, train_samples, width, label='Training Set')
    plt.bar(x + width / 2, test_samples, width, label='Test Set')

    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images across Classes')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_performance_metrics(model, test_set):
    """Plot F1 score, precision, recall, and ROC-AUC curves"""
    # Get predictions
    y_pred_proba = model.predict(test_set)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_set.classes
    class_names = list(test_set.class_indices.keys())

    # Create classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Plot precision, recall, f1-score for each class
    metrics_df = pd.DataFrame({
        'Precision': [report[cn]['precision'] for cn in class_names],
        'Recall': [report[cn]['recall'] for cn in class_names],
        'F1-Score': [report[cn]['f1-score'] for cn in class_names]
    }, index=class_names)

    plt.figure(figsize=(15, 6))
    metrics_df.plot(kind='bar', width=0.8)
    plt.title('Performance Metrics per Class')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_set):
    """Plot confusion matrix"""
    y_pred = np.argmax(model.predict(test_set), axis=1)
    y_true = test_set.classes
    class_names = list(test_set.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Add this code after your model training
import pandas as pd

print("Generating evaluation visualizations...")

# Plot training curves
plot_training_curves(history)

# Plot dataset distribution
plot_dataset_distribution(training_set, test_set)

# Plot performance metrics
plot_performance_metrics(model, test_set)

# Plot confusion matrix
plot_confusion_matrix(model, test_set)
# Part 3 - Making new predictions
##
##model_json=model.to_json()
##with open("model1.json", "w") as json_file:
##    json_file.write(model_json)
### serialize weights to HDF5
##    model.save_weights("model1.h5")
##    print("Saved model to disk")
