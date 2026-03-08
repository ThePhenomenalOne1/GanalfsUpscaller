import os
import torch
from pathlib import Path

def check_path(path_str):
    p = Path(path_str)
    if p.exists():
        if p.is_file():
            print(f"FILE: {p} | SIZE: {p.stat().st_size} bytes")
        elif p.is_dir():
            print(f"DIR : {p}")
            for item in p.iterdir():
                check_path(str(item))
    else:
        # print(f"NOT FOUND: {p}")
        pass

# 1. Check common torch hub locations
hub_dirs = [
    torch.hub.get_dir(),
    os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub'),
    os.path.join(os.environ.get('APPDATA', ''), 'torch', 'hub'),
]

print("Checking potential model locations...")
for d in hub_dirs:
    check_path(d)

# 2. Check current directory
check_path(".")

# 3. Check specific GFPGAN location
check_path("gfpgan/weights")
