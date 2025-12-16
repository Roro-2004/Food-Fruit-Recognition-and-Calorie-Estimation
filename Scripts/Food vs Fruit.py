import os
import cv2
import numpy as np
from tensorflow import keras

MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage1_FoodFruit\Final_model_inceptionresnetv2A_acc_0.9878.keras"
IMG_SIZE = (224, 224)
INPUT_FOLDER = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Test"
OUTPUT_TXT = "food_fruit_predictions.txt"
CLASSES = ["Food", "Fruit"]

def load_model():
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded - Food vs Fruit classifier")
    return model

def preprocess_image(img):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0).astype("float32")
    return img

def predict_image(model, img):
    img_input = preprocess_image(img)
    if img_input is None:
        return None, None
    
    pred = model.predict(img_input, verbose=0)[0][0]
    
    if pred > 0.5:
        predicted_class = CLASSES[1]
        confidence = pred
    else:
        predicted_class = CLASSES[0]
        confidence = 1 - pred
    
    return predicted_class, confidence

def process_images(model, input_folder):
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images")
    print("-" * 70)
    
    results = []
    for i, filename in enumerate(image_files, 1):
        try:
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read: {filename}")
                continue
            
            predicted_class, confidence = predict_image(model, img)
            
            if predicted_class is None:
                print(f"Failed to process: {filename}")
                continue
            
            display_img = img.copy()
            text = f"{predicted_class}: {confidence*100:.1f}%"
            cv2.putText(display_img, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Food vs Fruit Classifier", display_img)
            
            results.append({
                "image_name": filename,
                "predicted_class": predicted_class,
                "confidence": confidence
            })
            
            print(f"{i:3d}. {filename}")
            print(f"     Prediction: {predicted_class}")
            print(f"     Confidence: {confidence*100:.2f}%")
            print()
            
            key = cv2.waitKey(2000) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user")
                break
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    cv2.destroyAllWindows()
    return results

def print_results(results):
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if not results:
        print("No results to display")
        return
    
    for i, result in enumerate(results, 1):
        print(f"{i:3d}. {result['image_name']}")
        print(f"     Prediction: {result['predicted_class']}")
        print(f"     Confidence: {result['confidence']*100:.2f}%")
        print()

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result['image_name']}, {result['predicted_class']}, {result['confidence']*100:.2f}%\n")

def main():
    print("="*70)
    print("FOOD vs FRUIT CLASSIFIER")
    print("="*70)
    
    model = load_model()
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Folder not found - {INPUT_FOLDER}")
        return
    
    print(f"\nProcessing images from: {INPUT_FOLDER}")
    results = process_images(model, INPUT_FOLDER)
    
    if results:
        print_results(results)
        save_results(results, OUTPUT_TXT)
        print(f"Results saved to: {OUTPUT_TXT}")
    else:
        print("\nNo images processed")

if __name__ == "__main__":
    main()