
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Step 1: Load the IMDB dataset
num_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=num_words
)

# Step 2: Explore the dataset
print("Number of training samples:", len(x_train))
print("Number of test samples:", len(x_test))

print("Sample review (tokenized):", x_train[0])
print("Label (0 = Negative, 1 = Positive):", y_train[0])

# Step 3: Decode a sample review
word_index = imdb.get_word_index()

reverse_word_index = {
    value: key for key, value in word_index.items()
}

decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in x_train[0]]
)

print("\nDecoded Review:")
print(decoded_review)

# Step 4: Pad sequences
maxlen = 200

x_train = pad_sequences(
    x_train,
    maxlen=maxlen
)

x_test = pad_sequences(
    x_test,
    maxlen=maxlen
)

# Step 5: Define the LSTM model
model = models.Sequential([
    layers.Input(shape=(maxlen,)),
    layers.Embedding(
        input_dim=num_words,
        output_dim=32
    ),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid")
])

# Step 6: Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Step 7: Display model architecture
model.summary()

# Step 8: Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Step 9: Evaluate the model
test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=2
)

print("\nTest Accuracy:", test_acc)

# Step 10: Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")

# Loss plot
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")

plt.tight_layout()
plt.show()
