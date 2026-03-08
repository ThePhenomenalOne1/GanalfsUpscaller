import os
import torch
from pathlib import Path

def find_and_check_gfpgan():
    filename = "GFPGANv1.3.pth"
    hub_dir = torch.hub.get_dir()
    potential_paths = [
        os.path.join(hub_dir, 'checkpoints', filename),
        os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', filename),
        os.path.join('gfpgan', 'weights', filename)
    ]
    
    found = False
    for path in potential_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"Found {filename} at {path}")
            print(f"Size: {size} bytes")
            if size < 100000000: # If less than ~100MB, it's almost certainly corrupted
                print(f"File is too small ({size} bytes). Deleting it...")
                try:
                    os.remove(path)
                    print("Successfully deleted corrupted model file.")
                    found = True
                except Exception as e:
                    print(f"Error deleting file: {e}")
            else:
                print("File size looks correct, but it might still be corrupted. Deleting anyway to force re-download.")
                try:
                    os.remove(path)
                    print("Deleted model file to force re-download.")
                    found = True
                except Exception as e:
                    print(f"Error deleting file: {e}")
    
    if not found:
        print(f"Could not find {filename} in suspected locations.")

if __name__ == "__main__":
    find_and_check_gfpgan()
