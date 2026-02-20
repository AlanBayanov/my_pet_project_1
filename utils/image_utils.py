import cv2
import numpy as np

def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # Инверсия цветов
    return img.astype('float32') / 255.0