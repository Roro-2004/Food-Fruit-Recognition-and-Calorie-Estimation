import albumentations as A
import cv2
import numpy as np


def preprocess_image(img, target_size=(224, 224)):
   # Convert to RGB if not already RGB
    if len(img.shape) == 2:
     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Resize
    img = cv2.resize(img, target_size)
    # Sharpening
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    # Scale pixels to [0,1]
    img = img / 255.0

    return img

def get_augmentation_pipeline(target_size=(224,224)):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomResizedCrop(size=target_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)
    ])
    return transform

