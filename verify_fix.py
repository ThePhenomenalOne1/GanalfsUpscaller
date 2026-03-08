import sys
from pathlib import Path
import os
import cv2
import numpy as np

# Add project dir to path
sys.path.insert(0, str(Path(os.getcwd())))

from upscaler import upscale_image

def create_truncated_png(path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    size = os.path.getsize(path)
    with open(path, 'rb+') as f:
        f.truncate(size - 50) 

if __name__ == "__main__":
    test_path = "corrupted_verify.png"
    create_truncated_png(test_path)
    
    print(f"--- Testing {test_path} ---")
    try:
        # We don't need a real model for this test since it should fail during loading
        # but upscale_image calls get_upscaler which might try to download.
        # However, the loading check is BEFORE get_upscaler in upscale_image.
        upscale_image(test_path, "output_ignored.png")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected {type(e).__name__}: {e}")
    else:
        print("Error: upscale_image did not raise exception for corrupted file!")
        
    if os.path.exists(test_path):
        os.remove(test_path)
