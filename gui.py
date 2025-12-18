import gradio as gr
import os
import numpy as np
import cv2
from PIL import Image
from Integrated_Test_Pipeline import  Food_OR_Fruit_A,Food_Classifier_B,food_cal_map,Fruit_Classifier_C,fruit_cal_map,Binary_Segmentation_D,Multi_Segmentation_E


def analyze_image(input_img, filename="uploaded_100g.jpg"):
    # Save input to temp path to mimic your pipeline
    temp_path = "temp_gui_input.jpg"
    input_img_pil = Image.fromarray(input_img.astype('uint8'), 'RGB')
    input_img_pil.save(temp_path)

    # Extract grams (assume default 100g if not in name)
    grams = 100

    # Run pipeline logic
    foodORfruit_result = Food_OR_Fruit_A(temp_path)

    if foodORfruit_result == 'Food':
        food_class = Food_Classifier_B(temp_path)
        cal_per_g = food_cal_map.get(food_class, 0.0)
        total_cal = cal_per_g * grams
        result_text = f"Type: {foodORfruit_result}\nCategory: {food_class}\nCalories: {total_cal:.2f}"
        return result_text, None, None

    elif foodORfruit_result == 'Fruit':
        fruit_class = Fruit_Classifier_C(temp_path)
        cal_per_g = fruit_cal_map.get(fruit_class, 0.0)
        total_cal = cal_per_g * grams
        result_text = f"Type: {foodORfruit_result}\nCategory: {fruit_class}\nCalories: {total_cal:.2f}"

        binary_mask = Binary_Segmentation_D(temp_path)
        multi_mask = Multi_Segmentation_E(temp_path)

        # Convert to RGB for display
        binary_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        multi_rgb = multi_mask  # already RGB

        return result_text, binary_rgb, multi_rgb

    else:
        return "Unknown\nUnknown\n0.00", None, None

# ---  Gradio ---
with gr.Blocks(title="Food & Fruit Calorie Estimator") as demo:
    gr.Markdown("## 🍎 Food/Fruit Recognition & Calorie Estimation")
    gr.Markdown("Upload a food or fruit image to get its category, calories, and segmentation (for fruits).")

    with gr.Row():
        input_img = gr.Image(label="Upload Image", type="numpy")
        output_text = gr.Textbox(label="Result", lines=3)

    with gr.Row():
        binary_out = gr.Image(label="Binary Segmentation (Fruit)", interactive=False)
        multi_out = gr.Image(label="Multi-Class Segmentation (Fruit)", interactive=False)

    run_btn = gr.Button("Analyze")

    run_btn.click(
        fn=analyze_image,
        inputs=input_img,
        outputs=[output_text, binary_out, multi_out]
    )

    gr.Markdown("💡 **Note**: For accurate calorie estimation, include weight in filename like `apple_150g.jpg`.")

# Run
if __name__ == "__main__":
    demo.launch()