import os
import re

folder = r"D:\MOP\Amir Indesigns\کۆتاهاتووەکان\شاماران شاژنی ماران\شاماران شاژنی ماران\Links\upscaled"

count = 0
for name in os.listdir(folder):
    new_name = re.sub(r'_upscaled_4x(\.png)$', r'\1', name, flags=re.IGNORECASE)
    if new_name != name:
        old_path = os.path.join(folder, name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {name} -> {new_name}")
        count += 1

print(f"\nDone! Renamed {count} files.")
