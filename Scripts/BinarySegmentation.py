import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from glob import glob

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
    
    return image_tensor, image, (original_h, original_w)

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred_mask = model(image_tensor)
        pred_mask = (pred_mask > 0.5).float()
        return pred_mask.squeeze().cpu().numpy()

def resize_mask(mask, original_size):
    h, w = original_size
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def save_collage_predictions(image_paths, original_images, predictions, output_dir='predictions'):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image_path, original_img, mask) in enumerate(zip(image_paths, original_images, predictions)):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Convert binary mask to 3-channel grayscale 
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        collage = np.concatenate([original_img, mask_rgb], axis=1)
        collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"{base_name}_collage.png")
        cv2.imwrite(output_path, collage_bgr)
        

def main():
    MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Binary_Segmentation\best_train_unet_full_model (final).pth"
    TEST_IMAGES_DIR = r"C:\Users\dell\Downloads\Test Cases Structure\Test Cases Structure\Integerated Test"
    OUTPUT_DIR = "segmentation_results"
    IMG_SIZE = 256
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Test directory not found: {TEST_IMAGES_DIR}")
        return
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(TEST_IMAGES_DIR, ext)))
    
    if not image_paths:
        print("No images found in test directory.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH, device)
    
    all_original_images = []
    all_predictions = []
    processed_paths = []
    
    for image_path in image_paths:
        try:
            image_tensor, original_img, original_size = preprocess_image(image_path, IMG_SIZE)
            mask = predict_mask(model, image_tensor, device)
            mask_original = resize_mask(mask, original_size)
            
            all_original_images.append(original_img)
            all_predictions.append(mask_original)
            processed_paths.append(image_path)
            
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue
    
    if not all_original_images:
        print("No valid images processed.")
        return
    save_collage_predictions(processed_paths, all_original_images, all_predictions, OUTPUT_DIR)
        
if __name__ == "__main__":
    main()