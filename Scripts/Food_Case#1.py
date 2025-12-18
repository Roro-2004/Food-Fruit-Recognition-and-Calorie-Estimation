# Case 1 
import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from collections import defaultdict

TRAIN_ROOT = "C:\\Users\\dell\\Desktop\\Vision\\Project Data\\Food\\Train"
TEST_FOLDER = "C:\\Users\\dell\\Downloads\\Test Cases Structure\\Integerated Test"
EMBEDDING_WEIGHTS = "C:\\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\embedding.weights.h5"
OUTPUT_FILE = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\food_case1_output.txt"

EMBEDDINGS_CACHE = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\embeddings_cache\all_train_embeddings.npy"
PROTOTYPES_CACHE = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\embeddings_cache\class_prototypes.npy"
LABEL_TO_NAME_CACHE = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2-Food\embeddings_cache\label_to_name.json"

class L2Norm(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    return img

def load_dataset(root_dir):
    image_paths, labels = [], []
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for label, cat in enumerate(categories):
        cat_path = os.path.join(root_dir, cat)
        for img in os.listdir(cat_path):
            if img.lower().endswith(".jpg"):
                image_paths.append(os.path.join(cat_path, img))
                labels.append(label)

    return image_paths, labels, categories

def create_embedding_net():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    for layer in base.layers[:-27]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    embedding = L2Norm()(x)

    return Model(base.input, embedding)

embedding_net = create_embedding_net()
embedding_net.load_weights(EMBEDDING_WEIGHTS)

if os.path.exists(PROTOTYPES_CACHE) and os.path.exists(LABEL_TO_NAME_CACHE):
    with open(LABEL_TO_NAME_CACHE, 'r') as f:
        label_to_name = json.load(f) 
    label_to_name = {int(k): v for k, v in label_to_name.items()}
    train_class_prototypes = np.load(PROTOTYPES_CACHE, allow_pickle=True).item()
    
    print(f"Loaded {len(train_class_prototypes)} class prototypes from cache")
    print("Prototypes ready")
    
    train_paths, train_labels, train_categories = load_dataset(TRAIN_ROOT)
    
else:
    print("No cache found. Building prototypes from scratch...")
    train_paths, train_labels, train_categories = load_dataset(TRAIN_ROOT)
    label_to_name = {i: name for i, name in enumerate(train_categories)}
    
    def get_embedding(embedding_net, image_path):
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        emb = embedding_net.predict(img, verbose=0)
        return emb[0]
    
    def build_class_prototypes(embedding_net, paths, labels):
        class_dict = defaultdict(list)

        for p, l in zip(paths, labels):
            emb = get_embedding(embedding_net, p)
            class_dict[l].append(emb)

        class_prototypes = {}
        for cls, embs in class_dict.items():
            class_prototypes[cls] = np.mean(embs, axis=0)

        return class_prototypes
    
    train_class_prototypes = build_class_prototypes(embedding_net, train_paths, train_labels)
    print("Prototypes ready")


def get_embedding(embedding_net, image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    emb = embedding_net.predict(img, verbose=0)
    return emb[0]

def predict_food_class(image_path, threshold=0.9):
    emb = get_embedding(embedding_net, image_path)

    classes = list(train_class_prototypes.keys())
    prototypes = np.array([train_class_prototypes[cls] for cls in classes])
    
    # Compute distances vectorized
    distances = np.linalg.norm(emb - prototypes, axis=1)
    
    # Find minimum
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    best_class = classes[min_idx]

    if min_dist > threshold:
        return "Unknown food", min_dist
    else:
        return label_to_name[best_class], min_dist

test_images = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) if f.lower().endswith(".jpg")]

print(f"\nTesting {len(test_images)} images")

output_lines = []
output_lines.append(f"Testing {len(test_images)} images\n")

for img_path in test_images:
    pred_class, dist = predict_food_class(img_path, threshold=0.9)
    filename = os.path.basename(img_path)
    result_line = f"{filename} -> {pred_class} (dist: {dist:.4f})"
    print(result_line)  
    output_lines.append(result_line)  

# Save to file
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(output_lines))
