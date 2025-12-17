import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

MODEL_PATH = r"C:\Users\dell\Downloads\unet_fruit_segmentation.keras"
METADATA_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\MultiClass_Segmentation\model_metadata.json"
TEST_IMAGES_DIR = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Fruitssss"
OUTPUT_DIR = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\MultiClass_Segmentation\predictions"
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Load model metadata
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

FRUIT_CLASSES = metadata["classes"]
CLASS_TO_ID = metadata["class2id"]
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# Load color map from metadata
color_map = {}
for class_id_str, color in metadata["class_colors"].items():
    class_id = int(class_id_str)
    color_map[class_id] = color

# Ensure directories exist
for directory in [OUTPUT_DIR, MASKS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_and_preprocess(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    original_h, original_w = img.shape[:2]
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = img.astype("float32") / 255.0

    return original_img, img, (original_h, original_w)

def predict_mask(model, processed_img):
    input_tensor = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(input_tensor, verbose=0)[0]
    mask = np.argmax(prediction, axis=-1)
    return mask

def colorize_mask(mask, color_map):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored[mask == class_id] = color
    return colored

def get_detected_classes(mask, min_percentage=1.0):
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    detected = []
    
    for class_id, count in zip(unique, counts):
        if class_id == 0: 
            continue
            
        percentage = (count / total_pixels) * 100
        if percentage >= min_percentage:
            class_name = ID_TO_CLASS.get(class_id, f"Unknown_{class_id}")
            detected.append({
                'class': class_name,
                'percentage': float(percentage),
                'class_id': int(class_id)
            })
    
    return detected

def process_all_images():
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]

    if not image_files:
        print("No valid images found in test directory.")
        return []

    results = []
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        original_img, processed_img, original_size = load_and_preprocess(img_path)

        if original_img is None:
            continue

        small_mask = predict_mask(model, processed_img)
        mask = cv2.resize(small_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        colored_mask = colorize_mask(mask, color_map)
        
        detected_classes = get_detected_classes(mask)
        
        base_name = os.path.splitext(img_file)[0]

        np.save(os.path.join(MASKS_DIR, f"{base_name}_mask.npy"), mask)
        cv2.imwrite(
            os.path.join(MASKS_DIR, f"{base_name}_colored_mask.png"),
            cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
        )

        print(f"{img_file}: ", end="")
        if detected_classes:
            class_names = [dc['class'] for dc in detected_classes]
            print(", ".join(class_names))
        else:
            print("No fruits detected (or below 1% threshold)")

        results.append({
            'name': base_name,
            'original': original_img,
            'colored_mask': colored_mask,
            'mask': mask,
            'detected_classes': detected_classes
        })

    return results

def create_legend(color_map, id_to_class):
    fig, ax = plt.subplots(figsize=(8, max(10, len(color_map) * 0.3)))
    ax.axis('off')
    
    sorted_ids = sorted(color_map.keys())
    for idx, class_id in enumerate(sorted_ids):
        color = np.array(color_map[class_id]) / 255
        
        if class_id == 0:
            label = "Background"
        else:
            label = id_to_class.get(class_id, f"Class {class_id}")
        
        rect = plt.Rectangle((0, idx), 0.8, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
        ax.text(1.0, idx + 0.4, f"[{class_id}] {label}", va='center', fontsize=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(sorted_ids) - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    
    legend_path = os.path.join(VISUALIZATIONS_DIR, "color_legend.png")
    plt.savefig(legend_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_collages(results):
    for result in results:
        original = result['original']
        mask = result['colored_mask']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title(f"Original: {result['name']}", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(mask)
        axes[1].set_title("Predicted Segmentation Mask", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(VISUALIZATIONS_DIR, f"{result['name']}_collage.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    results = process_all_images()
    
    if results:
        create_legend(color_map, ID_TO_CLASS)
        save_collages(results)