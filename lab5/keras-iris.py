import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
# polega na odjęciu średniej wartości każdej cechy od wartości tej cechy w każdym wierszu danych, a następnie podzieleniu przez odchylenie standardowe tej cechy.
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
# Kodowanie „one hot” przekształca wartości kategoryczne na binarne kolumny, gdzie każda kolumna reprezentuje jedną kategorię. Wartość 1 oznacza obecność danej kategorii, a wartość 0 jej brak.
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), #liczba neuronów 4 (liczba cech)
    Dense(64, activation='relu'), #linear 100, tanh 97, elu 97.78, relu 100
    Dense(y_encoded.shape[1], activation='softmax') # liczba neuronów 3 (liczba klas)
])

# Compile the model
model.compile(optimizer='adam', loss='huber', metrics=['accuracy'])
#lion 100, ftrl 28, nadam 100, rmsprop 97,78
#dice 100, hinge 100, huber 100,
#możemy dostosować szybkośc uczenia się za pomocą np optimizer=Adam(learning_rate=0.0001), ale wpływa to na skutecznosc
# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size =16)
#zmiana batch size

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
