import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
#zmiana rozmiaru obrazu na 28x28
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
#zmiana labelów na kodowanie one hot
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix
# np.argmax zwraca indeksy maksymalnych wartości wzdłuż osi

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),# wejściem jest obraz 28x28x1 pikseli
    MaxPooling2D((2, 2)), #zmniejsze nie rozmiaru na 2x2
    Flatten(), #spłaszczenie macierzy do wektora
    Dense(64, activation='relu'), # wektor 64
    Dense(10, activation='softmax') #wektor 10 bo jest 10 cyfr
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
saver = ModelCheckpoint(filepath='Best_model.keras', save_best_only=True, save_freq='epoch') #zapisywnie modelu co epoch, tylko najlepszego
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history,saver])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy 9,4 7,2
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()
