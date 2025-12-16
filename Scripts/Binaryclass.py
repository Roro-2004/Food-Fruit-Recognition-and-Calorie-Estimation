import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob
import torch.serialization

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=256):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    image_resized = cv2.resize(image, (img_size, img_size))
    image_normalized = image_resized.astype('float32') / 255.0
    
    image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image, image_resized, (original_h, original_w)

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred_mask = model(image_tensor)
        pred_mask = (pred_mask > 0.5).float()
        return pred_mask.squeeze().cpu().numpy()

def resize_mask(mask, original_size):
    h, w = original_size
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def visualize_results(images_list, predictions_list, num_cols=3):
    num_images = len(images_list)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(15, 5 * num_rows))
    
    for idx in range(num_rows * num_cols):
        row = (idx // num_cols) * 2
        col = idx % num_cols
        
        if idx < num_images:
            axes[row, col].imshow(images_list[idx])
            axes[row, col].axis('off')
            
            axes[row + 1, col].imshow(predictions_list[idx], cmap='gray')
            axes[row + 1, col].axis('off')
        else:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    return fig

def apply_mask_to_image(image, mask, alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]
    return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

def save_predictions(image_paths, original_images, predictions, output_dir='predictions'):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image_path, original_img, mask) in enumerate(zip(image_paths, original_images, predictions)):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.png'), original_rgb)
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_mask.png'), mask_uint8)
        
        overlay = apply_mask_to_image(original_img, mask)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_overlay.png'), overlay_bgr)

def main():
    MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Binary_Segmentation\best_train_unet_full_model.pth"
    TEST_IMAGES_DIR = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\test_fruit"
    OUTPUT_DIR = "segmentation_results"
    IMG_SIZE = 256
    
    if not os.path.exists(TEST_IMAGES_DIR):
        return
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(TEST_IMAGES_DIR, ext)))
    
    if not image_paths:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model(MODEL_PATH, device)
        
        all_original_images = []
        all_predictions = []
        processed_paths = []
        
        for image_path in image_paths:
            try:
                image_tensor, original_img, _, original_size = preprocess_image(image_path, IMG_SIZE)
                mask = predict_mask(model, image_tensor, device)
                mask_original = resize_mask(mask, original_size)
                
                all_original_images.append(original_img)
                all_predictions.append(mask_original)
                processed_paths.append(image_path)
                
            except:
                continue
        
        if not all_original_images:
            return
        
        fig = visualize_results(all_original_images, all_predictions, num_cols=3)
        plt.show()
        
        save_predictions(processed_paths, all_original_images, all_predictions, OUTPUT_DIR)
        
    except:
        return

if __name__ == "__main__":
    main()