import sys
from pathlib import Path
import os

# Add project dir to path
sys.path.insert(0, str(Path(os.getcwd())))

from upscaler import get_upscaler
import torch

def test_model_config(model_alias):
    print(f"Testing alias: {model_alias}")
    model_map = {
        "realistic": "nomos8k_atd_jpg",
        "crisp": "4x-UltraSharp",
        "vivid": "RealESRGAN_x4plus_Vivid"
    }
    model_name = model_map[model_alias]
    
    # We don't want to actually download/load (too slow/might fail in CI)
    # but we can check if the logic for 'is_community_model' works if we mock the Path.exists
    
    print(f"  Model Name: {model_name}")
    is_community = any(name in model_name.lower() for name in ["remacri", "foolhardy", "ultrasharp", "nomos8k", "vivid"])
    print(f"  Is Community Model: {is_community}")
    
    if not is_community:
        print(f"  ERROR: {model_alias} should be a community model!")

if __name__ == "__main__":
    for alias in ["realistic", "crisp", "vivid"]:
        test_model_config(alias)
