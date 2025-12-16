import os
import cv2
import json
import numpy as np
from tensorflow import keras

MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2_Fruit\fruit_classifier_model.keras"
METADATA_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2_Fruit\fruit_classifier_model_metadata.json"
IMG_SIZE = (224, 224)
INPUT_FOLDER = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\test_fruit"
OUTPUT_TXT = "predictions_results.txt"

def load_model_and_metadata():
    model = keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"Model loaded - {len(metadata['classes'])} classes")
    return model, metadata

def preprocess_image(img):
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype("float32")
    return img

def predict_image(model, img, classes):
    img_input = preprocess_image(img)
    if img_input is None:
        return None, None
    
    preds = model.predict(img_input, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    predicted_class = classes[class_id]
    
    return predicted_class, confidence

def process_images(model, classes, input_folder):
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images")
    print("-" * 70)
    
    results = []
    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read: {filename}")
            continue
            
        predicted_class, confidence = predict_image(model, img, classes)
        
        if predicted_class is None:
            print(f"Failed to process: {filename}")
            continue
            
        display_img = img.copy()
        text = f"{predicted_class}: {confidence*100:.1f}%"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Prediction", display_img)
        
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
    
    cv2.destroyAllWindows()
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result['image_name']}, {result['predicted_class']}, {result['confidence']*100:.2f}%\n")

def main():
    print("="*70)
    print("FRUIT CLASSIFIER")
    print("="*70)
    
    model, metadata = load_model_and_metadata()
    classes = metadata["classes"]
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Folder not found - {INPUT_FOLDER}")
        return
    
    print(f"\nProcessing images from: {INPUT_FOLDER}")
    results = process_images(model, classes, INPUT_FOLDER)
    
    if results:
        save_results(results, OUTPUT_TXT)
        print(f"Results saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()