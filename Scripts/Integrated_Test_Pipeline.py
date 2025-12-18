import os
import re
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import tensorflow as tf
import torch
import json
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from collections import defaultdict


Test_Folder = r"Test Cases Structure/Integerated Test"
food_calories_path = r"C:\Users\menna\PycharmProjects\CV_Project\Project Data\Food\Train Calories.txt"
fruit_calories_path = r"C:\Users\menna\PycharmProjects\CV_Project\Project Data\Fruit\Calories.txt"

METADATA_PATH_part_C=r"fruit_classifier_model_metadata.json"
if not os.path.exists(METADATA_PATH_part_C):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH_part_C}")

with open(METADATA_PATH_part_C, "r") as f:
    CLASSES_part_C = json.load(f)["classes"]
########### METADATA_PATH_part_E
METADATA_PATH_part_E=r"model_metadata_part_E.json"
if not os.path.exists(METADATA_PATH_part_E):
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH_part_E}")

with open(METADATA_PATH_part_E, 'r') as f:
    _metadata_E = json.load(f)

# Build color map: {class_id (int) -> [R, G, B]}
COLOR_MAP_E = {}
for class_id_str, rgb in _metadata_E["class_colors"].items():
    COLOR_MAP_E[int(class_id_str)] = rgb  # Ensure key is int

output_root = os.path.join("Integrated_Results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(output_root, exist_ok=True)
#food recognition preparation


# Custom layer (required for loading)
@tf.keras.utils.register_keras_serializable()
class L2Norm(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def _build_embedding_net():
    """Recreate the exact architecture used in training."""
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

embedding_file = r"food recognition/embeddings_cache"
_MODEL_B = None
_CLASS_PROTOTYPES_B = None
_LABEL_TO_NAME_B = None
_WEIGHTS_PATH_B = r"food recognition/embedding.weights.h5"


# ======================
# Load model + cached prototypes
# ======================
def _load_model_and_prototypes():
    """Load model weights and cached prototypes (no image reprocessing)."""
    global _MODEL_B, _CLASS_PROTOTYPES_B, _LABEL_TO_NAME_B

    if _MODEL_B is not None:
        return  # already loaded

    # 1. Load model architecture + weights
    _MODEL_B = _build_embedding_net()
    if not os.path.exists(_WEIGHTS_PATH_B):
        raise FileNotFoundError(f"Weights not found: {_WEIGHTS_PATH_B}")
    _MODEL_B.load_weights(_WEIGHTS_PATH_B)
    print("✓ Model B loaded.")

    # 2. Load cached prototypes and label mapping
    protos_path = os.path.join(embedding_file, "class_prototypes.npy")
    labels_path = os.path.join(embedding_file, "label_to_name.json")

    if not os.path.exists(protos_path):
        raise FileNotFoundError(f"Prototypes file not found: {protos_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label mapping not found: {labels_path}")

    _CLASS_PROTOTYPES_B = np.load(protos_path, allow_pickle=True).item()
    with open(labels_path, "r", encoding="utf-8") as f:
        _LABEL_TO_NAME_B = json.load(f)

    # Ensure keys are integers (JSON saves dict keys as strings)
    _LABEL_TO_NAME_B = {int(k): v for k, v in _LABEL_TO_NAME_B.items()}

    print(f"✓ Loaded prototypes for {len(_CLASS_PROTOTYPES_B)} classes.")##################################################
def read_all_models():
    #Models pathes
    MODEL_Food_OR_Fruit_A = r"VisionModels/Final_model_inceptionresnetv2A_acc_0.9878.keras"
    # MODEL_Food_Classifier_B=
    MODEL_Fruit_Classifier_C = r"VisionModels/fruit_classifier_model.keras"

    MODEL_Binary_Segmentation_D = r"VisionModels/best_train_unet_full_model (final)D.pth"
    MODEL_Multi_Segmentation_E = r"VisionModels/unet_fruit_segmentation.keras"

    models_A= tf.keras.models.load_model(MODEL_Food_OR_Fruit_A, compile=False)


    models_C = tf.keras.models.load_model(MODEL_Fruit_Classifier_C, compile=False)
    models_D = torch.load(MODEL_Binary_Segmentation_D, map_location=torch.device('cpu'), weights_only=False)

    models_E = tf.keras.models.load_model(MODEL_Multi_Segmentation_E, compile=False)
    return models_A,models_C,models_D,models_E


def Food_OR_Fruit_A(img_path):
    CLASSES = ["Food", "Fruit"]
    #preprocess_image for part a
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMG_SIZE = (224, 224)
    img = cv2.resize(img, IMG_SIZE)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0).astype("float32")
    #predict
    pred = models_A.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        predicted_class = CLASSES[1]
    else:
        predicted_class = CLASSES[0]

    return predicted_class


def Food_Classifier_B(img_path):
    _load_model_and_prototypes()  # Ensure model & prototypes are ready

    # Preprocess input image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    # Get embedding
    emb = _MODEL_B.predict(img, verbose=0)[0]

    # Find class with minimum distance (always return the closest)
    min_dist = float("inf")
    best_class = None
    for cls, proto in _CLASS_PROTOTYPES_B.items():
        dist = np.linalg.norm(emb - proto)
        if dist < min_dist:
            min_dist = dist
            best_class = cls

    return _LABEL_TO_NAME_B[best_class]  # Always return the nearest class




def Fruit_Classifier_C(img_path):
    if models_C is None:
        raise RuntimeError("Model C is not loaded. Please load it first.")
    # Read and validate image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Preprocess:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_size=(224, 224)
    img_resized = cv2.resize(img_rgb, image_size)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_resized, -1, kernel)

    # Normalize and add batch dimension
    img_tensor = img_sharp.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # Predict
    preds = models_C.predict(img_tensor, verbose=0)[0]
    predicted_class = CLASSES_part_C[int(np.argmax(preds))]

    return predicted_class

def Binary_Segmentation_D(img_path):
    if models_D is None:
        raise RuntimeError("Model D is not loaded. Please load it first.")

    # Read and validate image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    original_h, original_w = image.shape[:2]

    # Preprocess:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to('cpu')

    # Predict mask
    with torch.no_grad():
        pred = models_D(image_tensor)
        pred = (pred > 0.5).float()  # Binary: 0.0 or 1.0

    # Resize to original size
    mask_small = pred.squeeze().cpu().numpy()  # (256, 256) with 0.0/1.0
    mask_original = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Convert to uint8 grayscale: 0 (black) and 255 (white)
    mask_uint8 = (mask_original * 255).astype(np.uint8)

    return mask_uint8  # Shape: (H, W), dtype: uint8, values: 0 or 255


def Multi_Segmentation_E(img_path):
    if models_E is None:
        raise RuntimeError("Model E is not loaded. Please load it first.")
    # --- Load & preprocess image ---
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
    original_h, original_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Handle grayscale/RGBA
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    # Resize + sharpen + normalize (match training preprocessing)
    img_resized = cv2.resize(img_rgb, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_resized, -1, kernel)
    img_tensor = np.expand_dims(img_sharp.astype(np.float32) / 255.0, axis=0)
    # --- Predict ---
    pred = models_E.predict(img_tensor, verbose=0)[0]  # (H, W, num_classes)
    small_mask = np.argmax(pred, axis=-1)  # (224, 224)
    # --- Resize mask to original size ---
    mask_original = cv2.resize(
        small_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )  # (H, W) with class IDs as integers
    # --- Colorize mask using metadata color map ---
    h, w = mask_original.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color_rgb in COLOR_MAP_E.items():
        colored_mask[mask_original == class_id] = color_rgb  # RGB order

    return colored_mask  # RGB image


def read_calories_mapping(calories_file_path):
    mapping = {}
    with open(calories_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            try:
                name, cal_part = line.split(':', 1)
                name = name.strip()
                cal_part = cal_part.strip().replace('~', '').strip()
                # Remove "calories per gram" and any trailing non-number text
                cal_part = cal_part.split()[0]  # Take only the first word (the number)
                cal_val = float(cal_part)
                mapping[name] = cal_val
            except Exception:
                continue
    return mapping
def extract_grams_from_filename(filename):
    match = re.search(r'_(\d+)g\.', filename)
    if match:
        return int(match.group(1))
    return 100


def Pipeline(img_path):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"------------------{base_name}------------------")
    result_dir = os.path.join(output_root, base_name)
    os.makedirs(result_dir, exist_ok=True)

    grams = extract_grams_from_filename(os.path.basename(img_path))
    foodORfruit_result = Food_OR_Fruit_A(img_path)



    line1 = foodORfruit_result
    line2 = ""
    line3 = ""

    if foodORfruit_result == 'Food':
        food_class = Food_Classifier_B(img_path)
        line2 = food_class
        cal_per_g = food_cal_map.get(food_class, 0.0)
        total_cal = cal_per_g * grams
        line3 = f"{total_cal:.2f}"

    elif foodORfruit_result == 'Fruit':
        fruit_class = Fruit_Classifier_C(img_path)
        line2 = fruit_class
        cal_per_g = fruit_cal_map.get(fruit_class, 0.0)
        total_cal = cal_per_g * grams
        line3 = f"{total_cal:.2f}"

        binary_mask = Binary_Segmentation_D(img_path)
        binary_save_path = os.path.join(result_dir, f"{base_name}_binary_mask.png")
        cv2.imwrite(binary_save_path, binary_mask)

        multi_mask = Multi_Segmentation_E(img_path)
        multi_save_path = os.path.join(result_dir, f"{base_name}_multiclass_mask.png")
        cv2.imwrite(multi_save_path, multi_mask)
    else:
        line1 = "Unknown"
        line2 = "Unknown"
        line3 = "0.00"

    print(line1)
    print(line2)
    print(line3)


    txt_path = os.path.join(result_dir, f"{base_name}_result.txt")
    with open(txt_path, 'w') as f:
        f.write(f"{line1}\n{line2}\n{line3}\n")

################################################################
food_cal_map = read_calories_mapping(food_calories_path)
fruit_cal_map = read_calories_mapping(fruit_calories_path)
# print(food_cal_map)
# print(fruit_cal_map)
models_A,models_C, models_D, models_E= read_all_models()
for filename in os.listdir(Test_Folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(Test_Folder, filename)
        Pipeline(img_path)
