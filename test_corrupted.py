import cv2
import numpy as np
from PIL import Image
import os

def create_truncated_png(path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    size = os.path.getsize(path)
    with open(path, 'rb+') as f:
        f.truncate(size - 50) 

def test_loading(path):
    print(f"Testing {path}...")
    try:
        img = cv2.imread(path)
        if img is None:
            print("cv2.imread returned None")
        else:
            print("cv2.imread succeeded")
    except Exception as e:
        print(f"cv2.imread raised: {e}")

    try:
        with Image.open(path) as pil_img:
            # Forcing load to trigger EOF
            pil_img.load()
            print("PIL.Image.load succeeded")
    except Exception as e:
        print(f"PIL.Image.load raised: {e}")

    try:
        with Image.open(path) as pil_img:
            exif = pil_img.getexif()
            print("PIL.Image.getexif succeeded")
    except Exception as e:
        print(f"PIL.Image.getexif raised: {e}")

if __name__ == "__main__":
    test_path = "corrupted_test.png"
    create_truncated_png(test_path)
    test_loading(test_path)
    if os.path.exists(test_path):
        os.remove(test_path)
