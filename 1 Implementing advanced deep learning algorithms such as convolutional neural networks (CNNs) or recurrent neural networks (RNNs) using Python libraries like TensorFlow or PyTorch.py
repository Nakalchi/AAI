
# Import Required Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Step 1: Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Step 2: Reduce Dataset Size for Faster Training
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]


# Step 3: Normalize Data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Step 4: Build CNN Model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Step 5: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Step 6: Train Model
print("Starting CNN Training...")

history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)


# Step 7: Evaluate Model
test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print("\nTest Accuracy:", test_acc)


# Step 8: Plot Accuracy
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show()


# Step 9: Save Model
model.save("cifar10_cnn_model.keras")

print("\nModel saved successfully!")
