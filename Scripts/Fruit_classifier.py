import os
import cv2
import json
import numpy as np
from tensorflow import keras

MODEL_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2_Fruit\fruit_classifier_model.keras"
METADATA_PATH = r"C:\Users\dell\Desktop\Food-Fruit-Recognition-and-Calorie-Estimation\Stage2_Fruit\fruit_classifier_model_metadata.json"
IMG_SIZE = (224, 224)
INPUT_FOLDER = r"C:\Users\dell\Downloads\Test Cases Structure\Test Cases Structure\Integerated Test"
OUTPUT_TXT = "Fruit_predictions.txt"

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
        
        results.append({
            "image_name": filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    
    cv2.destroyAllWindows()
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result['image_name']}, {result['predicted_class']}, {result['confidence']*100:.2f}%\n")

def print_results(results):
    print("PREDICTION RESULTS")
    
    if not results:
        print("No results to display")
        return
    
    for i, result in enumerate(results, 1):
        print(f"{i:3d}. {result['image_name']}")
        print(f"     Prediction: {result['predicted_class']}")
        print(f"     Confidence: {result['confidence']*100:.2f}%")
        
def main():
    print("FRUIT CLASSIFIER")
    model, metadata = load_model_and_metadata()
    classes = metadata["classes"]
    results = process_images(model, classes, INPUT_FOLDER)    
    if results:
        print_results(results)
        save_results(results, OUTPUT_TXT)
        print(f"Results saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()