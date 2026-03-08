import os
import torch
from pathlib import Path

# Common locations for torch hub
hub_dir = torch.hub.get_dir()
print(f"Torch Hub Dir: {hub_dir}")

checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
print(f"Checkpoints Dir: {checkpoints_dir}")

if os.path.exists(checkpoints_dir):
    print("Files in checkpoints:")
    for f in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size} bytes")
else:
    print("Checkpoints directory does not exist.")

# Also check local project
local_gfpgan = Path("gfpgan/weights")
if local_gfpgan.exists():
    print("Files in local gfpgan/weights:")
    for f in local_gfpgan.iterdir():
        print(f"  {f.name}: {f.stat().st_size} bytes")
