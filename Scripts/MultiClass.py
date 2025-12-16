import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\MultiClass_Segmentation\unet_fruit_segmentation.keras"
TEST_IMAGES_DIR = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\test_fruit"
OUTPUT_DIR = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\MultiClass_Segmentation\predictions"
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

FRUIT_CLASSES = [
    "Apple", "Banana", "Orange", "Grape", "Strawberry",
    "Pineapple", "Watermelon", "Mango", "Kiwi", "Peach"
]

for directory in [OUTPUT_DIR, MASKS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

np.random.seed(42)
num_classes = len(FRUIT_CLASSES) + 1
color_map = {0: [0, 0, 0]}

for i, fruit in enumerate(FRUIT_CLASSES):
    hue = i / len(FRUIT_CLASSES)
    r = int(255 * (0.5 + 0.5 * np.sin(hue * 2 * np.pi)))
    g = int(255 * (0.5 + 0.5 * np.sin(hue * 2 * np.pi + 2 * np.pi / 3)))
    b = int(255 * (0.5 + 0.5 * np.sin(hue * 2 * np.pi + 4 * np.pi / 3)))
    color_map[i + 1] = [r, g, b]

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

def process_all_images():
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for file in os.listdir(TEST_IMAGES_DIR):
        if os.path.splitext(file.lower())[1] in valid_extensions:
            image_files.append(file)
    
    if not image_files:
        return []
    
    all_results = []
    
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        original_img, processed_img, original_size = load_and_preprocess(img_path)
        
        if original_img is None:
            continue
        
        small_mask = predict_mask(model, processed_img)
        mask = cv2.resize(small_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        colored_mask = colorize_mask(mask, color_map)
        
        base_name = os.path.splitext(img_file)[0]
        mask_npy_path = os.path.join(MASKS_DIR, f"{base_name}_mask.npy")
        np.save(mask_npy_path, mask)
        
        mask_png_path = os.path.join(MASKS_DIR, f"{base_name}_mask.png")
        cv2.imwrite(mask_png_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        
        all_results.append({
            'name': base_name,
            'original': original_img,
            'mask': mask,
            'colored_mask': colored_mask
        })
    
    return all_results

def create_legend(color_map, fruit_classes):
    fig, ax = plt.subplots(figsize=(4, max(3, len(color_map) * 0.3)))
    ax.axis('off')
    
    for class_id, color in sorted(color_map.items()):
        if class_id == 0:
            name = "Background"
        else:
            name = fruit_classes[class_id - 1]
        
        rect = plt.Rectangle((0, class_id), 0.8, 0.8, color=np.array(color) / 255)
        ax.add_patch(rect)
        ax.text(1, class_id + 0.4, f"{name}", va='center', fontsize=9)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(color_map) + 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig

def plot_all_results(results, color_map, fruit_classes, max_cols=3):
    if not results:
        return
    
    num_images = len(results)
    num_cols = min(max_cols, num_images)
    num_rows = int(np.ceil(num_images / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(num_cols * 6, num_rows * 3))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    current_axis = 0
    
    for i, result in enumerate(results):
        col_start = (i % num_cols) * 2
        
        if num_rows > 1:
            row_idx = i // num_cols
            axes_original = axes[row_idx, col_start]
            axes_mask = axes[row_idx, col_start + 1]
        else:
            axes_original = axes_flat[col_start]
            axes_mask = axes_flat[col_start + 1]
        
        axes_original.imshow(result['original'])
        axes_original.set_title(f"{result['name']}", fontsize=10)
        axes_original.axis('off')
        
        axes_mask.imshow(result['colored_mask'])
        axes_mask.set_title("Mask", fontsize=10)
        axes_mask.axis('off')
        
        current_axis += 2
    
    for j in range(current_axis, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(VISUALIZATIONS_DIR, "all_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    legend_fig = create_legend(color_map, fruit_classes)
    legend_path = os.path.join(VISUALIZATIONS_DIR, "color_legend.png")
    legend_fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(legend_fig)
    
    return fig

if __name__ == "__main__":
    results = process_all_images()
    
    if results:
        fig = plot_all_results(results, color_map, FRUIT_CLASSES)
        plt.show()
