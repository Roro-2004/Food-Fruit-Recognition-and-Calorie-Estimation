#CASE 2
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import register_keras_serializable

TEST_FOLDER = "C:\\Users\\dell\\Downloads\\Test Cases Structure\\Siamese Case II Test"
EMBEDDING_WEIGHTS = "C:\\Users\\dell\\Desktop\\Food-Fruit-Recognition-and-Calorie-Estimation\\Stage2-Food\\embedding.weights.h5"
OUTPUT_FILE = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\food_case2_output.txt"

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

@register_keras_serializable()
class L2Norm(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def create_embedding_net():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    for layer in base.layers[:-27]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    embedding = L2Norm()(x)
    return Model(base.input, embedding)

embedding_net = create_embedding_net()
embedding_net.load_weights(EMBEDDING_WEIGHTS)


all_images = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.lower().endswith(".jpg")]

anchor_path = None
reference_paths = []

for path in all_images:
    if "anchor" in os.path.basename(path).lower():
        anchor_path = path
    else:
        reference_paths.append(path)

assert anchor_path is not None, "Anchor image not found in the folder!"

anchor_emb = embedding_net.predict(preprocess_image(anchor_path), verbose=0)

reference_embeddings = {}
for ref_path in reference_paths:
    reference_embeddings[ref_path] = embedding_net.predict(preprocess_image(ref_path), verbose=0)


distances = []

for ref_path, ref_emb in reference_embeddings.items():
    dist = np.linalg.norm(anchor_emb - ref_emb)
    distances.append((os.path.basename(ref_path), dist))

distances.sort(key=lambda x: x[1])

threshold = 0.9
closest_name, min_dist = distances[0]

output = ""
output += f"Anchor image: {os.path.basename(anchor_path)}\n\n"
output += "Distances to reference images:\n"
for name, dist in distances:
    output += f"{name}: {dist:.4f}\n"

if min_dist > threshold:
    output += "\nNo match found for the anchor image."
else:
    output += f"\nMost similar image: {closest_name} (Distance: {min_dist:.4f})"
print(output)

# Save to file
with open(OUTPUT_FILE, 'w') as f:
    f.write(output)