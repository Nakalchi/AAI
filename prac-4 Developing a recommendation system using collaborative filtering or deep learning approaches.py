
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess the dataset

file_path = r"E:\tkt\recsys-movielens100k-main\recsys-movielens100k-main\benchmark\data\u.data"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found at: {file_path}"
    )

column_names = [
    "user_id",
    "item_id",
    "rating",
    "timestamp"
]

data = pd.read_csv(
    file_path,
    sep="\t",
    names=column_names
)

print("Dataset loaded successfully!")
print(data.head())

# Normalize user IDs and item IDs to start from 0
data["user_id"] = data["user_id"] - 1
data["item_id"] = data["item_id"] - 1

# Extract number of users and items
num_users = int(data["user_id"].max() + 1)
num_items = int(data["item_id"].max() + 1)

print("\nNumber of users:", num_users)
print("Number of items:", num_items)

# Step 2: Split data into training and testing

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Step 3: Create TensorFlow datasets

def create_tf_dataset(df):

    users = tf.constant(
        df["user_id"].values,
        dtype=tf.int32
    )

    items = tf.constant(
        df["item_id"].values,
        dtype=tf.int32
    )

    ratings = tf.constant(
        df["rating"].values,
        dtype=tf.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        ((users, items), ratings)
    )

    return dataset.shuffle(1024).batch(32)


train_dataset = create_tf_dataset(train_data)
test_dataset = create_tf_dataset(test_data)

# Step 4: Define the Recommendation Model

class MatrixFactorizationModel(tf.keras.Model):

    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=int(num_users),
            output_dim=int(embedding_dim)
        )

        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=int(num_items),
            output_dim=int(embedding_dim)
        )

    def call(self, inputs):

        user_vector = self.user_embedding(inputs[0])

        item_vector = self.item_embedding(inputs[1])

        dot_product = tf.reduce_sum(
            user_vector * item_vector,
            axis=1
        )

        return dot_product


model = MatrixFactorizationModel(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=50
)

# Step 5: Compile the model

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01
    ),
    loss="mse",
    metrics=["mae"]
)

# Step 6: Train the model

print("\nTraining the recommendation model...")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

# Step 7: Evaluate the model

test_loss, test_mae = model.evaluate(
    test_dataset,
    verbose=2
)

print("\nModel Evaluation:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Step 8: Plot training history

plt.figure(figsize=(12, 4))

# Loss graph
plt.subplot(1, 2, 1)

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
plt.title("Loss over Epochs")

# MAE graph
plt.subplot(1, 2, 2)

plt.plot(
    history.history["mae"],
    label="Train MAE"
)

plt.plot(
    history.history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.title("MAE over Epochs")

plt.tight_layout()
plt.show()

# Step 9: Make movie recommendations

def recommend(user_id, top_k=5):

    if user_id < 0 or user_id >= num_users:
        raise ValueError("Invalid user ID!")

    # Predict ratings for all movies
    user_vector = tf.constant(
        [user_id] * num_items,
        dtype=tf.int32
    )

    item_vector = tf.constant(
        list(range(num_items)),
        dtype=tf.int32
    )

    predictions = model.predict(
        (user_vector, item_vector),
        verbose=0
    ).flatten()

    # Exclude movies already rated by the user
    rated_items = data[
        data["user_id"] == user_id
    ]["item_id"].values

    predictions[rated_items] = -np.inf

    # Get top recommended movies
    top_items = np.argsort(-predictions)[:top_k]

    return top_items


# Step 10: Display recommendations

user_id = 0

recommended_items = recommend(
    user_id=user_id,
    top_k=5
)

print(f"\nRecommended movies for User {user_id + 1}:")

for item in recommended_items:
    print(f"Movie ID: {item + 1}")

print("\nRecommendation system completed successfully!")
