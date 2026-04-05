"""
Local Image Upscaler using Real-ESRGAN
The best free, open-source AI image upscaling solution.

Features:
- 2x, 4x upscaling
- Works with photos and anime/illustrations
- Face enhancement option (GFPGAN)
- Batch processing support
- GPU acceleration (if available)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import psutil

# Memory limit in GB
MAX_MEMORY_GB = 10

# Real-ESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def add_film_grain(image, strength=0.5):
    """
    Add subtle film grain noise to make images look more natural.
    
    Args:
        image: numpy array (H, W, C) in BGR format
        strength: noise strength (0.0 to 1.0, recommended 0.3-0.7)
    
    Returns:
        Image with added grain
    """
    if strength <= 0:
        return image
    
    # Create noise pattern
    noise = np.random.normal(0, strength * 5, image.shape).astype(np.float32)
    
    # Apply noise
    noisy_image = image.astype(np.float32) + noise
    
    # Clip to valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def apply_sharpening(image, amount=0.5):
    """
    Apply unsharp mask to the image to increase perceived sharpness.
    
    Args:
        image: numpy array (BGR)
        amount: sharpening strength (0.0 to 1.0)
    """
    if amount <= 0:
        return image
        
    # Unsharp mask parameters
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, gaussian_blur, -amount, 0)
    
    return sharpened



def download_models():
    """Download required models if not present."""
    import urllib.request
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Use reliable mirrors and set a User-Agent to avoid blocks
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)
    
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        # Specialized cartoon/illustration model
        "4x_foolhardy_Remacri.pth": "https://github.com/styler00number/Model-Zoo/releases/download/models/4x_foolhardy_Remacri.pth",
        # Realistic models
        "4x-UltraSharp.pth": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"📥 Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                
                # Check if we actually got a binary file by looking at its size (error pages are small)
                if model_path.exists() and model_path.stat().st_size < 5000:
                    print(f"❌ Downloaded file for {model_name} is too small, likely an error page.")
                    model_path.unlink()
                    return False
                    
                print(f"✅ Downloaded {model_name}")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                if model_path.exists():
                    model_path.unlink()
                return False
    return True


def check_memory_usage():
    """Check current memory usage and return available memory in GB."""
    process = psutil.Process()
    used_gb = process.memory_info().rss / (1024 ** 3)
    return used_gb


def enforce_memory_limit():
    """Enforce memory limit and return whether we're within limits."""
    current_usage = check_memory_usage()
    if current_usage > MAX_MEMORY_GB:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return False
    return True


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# Global model cache to avoid reloading
_model_cache = {}


def load_image_robustly(path, progress_callback=None):
    """
    Load an image robustly, handling potential corruption or truncation.
    
    Args:
        path: Path to the image file
    
    Returns:
        tuple: (numpy_image, error_message)
    """
    path_str = str(path)
    
    # Check if file exists and has content
    if not os.path.exists(path_str):
        return None, f"File not found: {path_str}"
    
    if os.path.getsize(path_str) == 0:
        return None, "File is empty (0 bytes)"

    # 1. Try validation with PIL first (best for detecting truncation/EOF)
    try:
        from PIL import Image, ImageFile
        # Allow loading of truncated images if needed, but we want to know about it
        ImageFile.LOAD_TRUNCATED_IMAGES = False 
        
        with Image.open(path_str) as img:
            img.verify() # This checks for corruption without loading pixels fully
            
        # Re-open to actually load, as verify() can close the file/invalidate the object
        with Image.open(path_str) as img:
            img.load() # Force pixel loading to check for decompression errors
            
    except (IOError, EOFError, ValueError) as e:
        error_msg = str(e)
        if "unexpected EOF" in error_msg.lower() or "truncated" in error_msg.lower():
            return None, f"Image file is corrupted or truncated: {error_msg}"
        return None, f"Failed to validate image with PIL: {error_msg}"
    except Exception as e:
        return None, f"Unexpected error validating image: {e}"

    # 2. Try loading with OpenCV (standard for this upscaler)
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        # Fallback: Load with PIL and convert to OpenCV format
        try:
            with Image.open(path_str) as pil_img:
                # Convert to RGB then BGR
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                if progress_callback:
                    progress_callback("Loaded via PIL fallback")
        except Exception as e:
            return None, f"OpenCV failed and PIL fallback also failed: {e}"

    return img, None


class PreloadedRealESRGANer(RealESRGANer):
    """
    A modified RealESRGANer that cleanly bypasses the internal initialization.
    The original library crashes with AttributeError on `.startswith` if model_path=None.
    """
    def __init__(self, scale, model, tile=0, tile_pad=10, pre_pad=10, half=False):
        import torch
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.model.eval()
        if self.half:
            self.model = self.model.half()


def get_upscaler(model_name: str = "RealESRGAN_x4plus", scale: int = 4, half_precision: bool = True, progress_callback=None):
    """
    Initialize the Real-ESRGAN upscaler.
    
    Args:
        model_name: Model to use ('RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus')
        scale: Upscaling factor (2 or 4)
        half_precision: Use FP16 for faster processing (GPU only)
    
    Returns:
        RealESRGANer instance
    """
    global _model_cache
    
    # Check cache first
    cache_key = f"{model_name}_{scale}_{half_precision}"
    if cache_key in _model_cache:
        if progress_callback:
            progress_callback("Using cached model (fast)")
        return _model_cache[cache_key]
    
    if progress_callback:
        progress_callback("Step 1/5: Checking model files...")
    
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        if progress_callback:
            progress_callback("Downloading model (may take a few minutes)...")
        print("Models not found. Attempting to download...")
        if not download_models():
            raise FileNotFoundError(f"Missing model file: {model_name}.pth. The automatic download failed. Please check your internet connection or download it manually into the 'models' folder.")
    
    # Final check
    if not model_path.exists():
         raise FileNotFoundError(f"Missing model file: {model_name}.pth after download attempt.")
    
    if progress_callback:
        progress_callback("Step 2/5: Configuring model architecture...")
    
    # Configure model architecture based on model type
    is_community_model = any(name in model_name.lower() for name in ["remacri", "foolhardy", "ultrasharp"])
    
    if "anime" in model_name.lower():
        # Anime model has different architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif "x2plus" in model_name.lower():
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
    else:
        # Default x4plus architecture (works for most community models too)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    if progress_callback:
        progress_callback("Step 3/5: Initializing GPU/CPU...")
    
    # Check for GPU availability
    import torch
    gpu_available = torch.cuda.is_available()
    
    # Apply memory limit
    if gpu_available:
        # Limit GPU memory to MAX_MEMORY_GB
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb > MAX_MEMORY_GB:
            fraction = MAX_MEMORY_GB / total_vram_gb
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)} (limited to {MAX_MEMORY_GB}GB)")
        else:
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f}GB available)")
        if progress_callback:
            progress_callback(f"Step 3/5: Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"💻 Using CPU (limited to {MAX_MEMORY_GB}GB RAM)")
        if progress_callback:
            progress_callback("Step 3/5: Using CPU (this will be slower)")
        half_precision = False  # CPU doesn't support half precision
    
    if progress_callback:
        progress_callback("Step 4/5: Loading AI model weights (this is the slow part)...")
    
    # Use tiling for memory-constrained processing - always use tiling to prevent OOM
    tile_size = 400  # Always use tiling for stability
    
    # For community models, we need to load weights manually
    # because they use a different state dict format
    if is_community_model:
        import torch
        
        if progress_callback:
            progress_callback(f"Loading community model ({model_name})...")
        
        # Load the state dict manually
        loadnet = torch.load(str(model_path), map_location=torch.device('cpu'), weights_only=False)
        
        # Debug: print first few keys
        if isinstance(loadnet, dict):
            sample_keys = list(loadnet.keys())[:5]
            print(f"Model top-level keys: {sample_keys}")
        
        # Check if weights are nested under 'params' or 'params_ema' or stored directly
        if isinstance(loadnet, dict):
            if 'params_ema' in loadnet:
                state_dict = loadnet['params_ema']
            elif 'params' in loadnet:
                state_dict = loadnet['params']
            elif 'model' in loadnet:
                state_dict = loadnet['model']
            else:
                # Weights stored directly (community model format)
                state_dict = loadnet
        else:
            state_dict = loadnet
        
        # Debug: print sample weight keys
        if isinstance(state_dict, dict):
            sample_keys = list(state_dict.keys())[:5]
            print(f"State dict sample keys: {sample_keys}")
        
        # Try to load weights - use non-strict loading for community models
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded model with strict=True")
        except RuntimeError as e:
            print(f"Strict loading failed, trying key remapping...")
            
            # Precise key remapping for legacy ESRGAN architectures (like 4x-UltraSharp)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                
                # Handling RRDB blocks: model.1.sub.[block_idx].RDB[sub_idx].conv[c_idx].0.[weight|bias]
                # -> body.[block_idx].rdb[sub_idx].conv[c_idx].[weight|bias]
                if k.startswith('model.1.sub.'):
                    parts = k.split('.')
                    if len(parts) >= 8 and parts[4].startswith('RDB') and parts[5].startswith('conv'):
                        block_idx = int(parts[3])
                        rdb_idx = parts[4].replace('RDB', 'rdb') # e.g., 'rdb1'
                        conv_idx = parts[5] # e.g., 'conv1'
                        wb = parts[7] # 'weight' or 'bias'
                        new_key = f'body.{block_idx}.{rdb_idx}.{conv_idx}.{wb}'
                    elif len(parts) == 6 and parts[3] == '23': # model.1.sub.23.weight
                        new_key = f'conv_body.{parts[5]}'
                elif k.startswith('model.0.'):
                    new_key = f'conv_first.{k.split(".")[2]}'
                elif k.startswith('model.3.'):
                    new_key = f'conv_up1.{k.split(".")[2]}'
                elif k.startswith('model.6.'):
                    new_key = f'conv_up2.{k.split(".")[2]}'
                elif k.startswith('model.8.'):
                    new_key = f'conv_hr.{k.split(".")[2]}'
                elif k.startswith('model.10.'):
                    new_key = f'conv_last.{k.split(".")[2]}'
                
                new_state_dict[new_key] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print("Loaded model with key remapping (strict=False)")
            except Exception as e3:
                raise RuntimeError(f"Failed to load community model weights: {e3}")
        
        model.eval()
        if gpu_available:
            model = model.to(torch.device('cuda'))
        
        # Use our custom RealESRGANer subclass with pre-loaded model
        # The original class throws AttributeError on NoneType if model_path=None
        upsampler = PreloadedRealESRGANer(
            scale=scale,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=half_precision and gpu_available
        )
    else:
        # Standard Real-ESRGAN model loading
        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=tile_size,  # Use tiling for stability
            tile_pad=10,
            pre_pad=0,
            half=half_precision and gpu_available
        )
    
    if progress_callback:
        progress_callback("Step 5/5: Model loaded successfully!")
    
    # Cache for future use
    _model_cache[cache_key] = upsampler
    
    return upsampler


def upscale_image(
    input_path: str,
    output_path: str = None,
    model_name: str = "RealESRGAN_x4plus",
    scale: int = 4,
    face_enhance: bool = False,
    output_format: str = "png",
    progress_callback=None,
    dpi: int = 72,
    suffix: str = "_upscaled",
    preserve_metadata: bool = True,
    remove_watermark: bool = False,
    grain_strength: float = 0.0,
    sharpen_amount: float = 0.0
) -> str:
    """
    Upscale a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path for output (auto-generated if None)
        model_name: Model to use
        scale: Upscaling factor
        face_enhance: Enable face enhancement with GFPGAN
        output_format: Output format (png, jpg, webp)
        dpi: Output DPI metadata
        suffix: Custom suffix for output filename (default: '_upscaled')
        preserve_metadata: Copy EXIF metadata from source image
    
    Returns:
        Path to the upscaled image
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Generate output path if not specified
    if output_path is None:
        output_dir = input_path.parent / "upscaled"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}.{output_format}"
    
    output_path = Path(output_path)
    
    print(f"🖼️  Processing: {input_path.name}")
    
    # Read image robustly
    img, error = load_image_robustly(input_path, progress_callback=progress_callback)
    if error:
        raise ValueError(error)
    
    if img is None:
        raise ValueError(f"Failed to read image (unknown error): {input_path}")
    
    original_size = f"{img.shape[1]}x{img.shape[0]}"
    
    # Remove watermark if enabled
    if remove_watermark:
        try:
            from watermark_removal import process_watermark_removal
            img, watermark_found = process_watermark_removal(img, enabled=True)
            if watermark_found:
                print("✨ Watermark detected and removed")
                if progress_callback:
                    progress_callback("Watermark removed")
        except Exception as e:
            print(f"⚠️  Watermark removal failed: {e}")
    
    # Get upscaler with progress updates
    upsampler = get_upscaler(model_name, scale, progress_callback=progress_callback)
    
    # Face enhancer (optional)
    face_enhancer = None
    if face_enhance:
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
            print("👤 Face enhancement enabled")
        except ImportError:
            print("⚠️  GFPGAN not installed. Skipping face enhancement.")
    
    # Enforce memory limit before processing
    if not enforce_memory_limit():
        print(f"⚠️  Memory usage exceeded {MAX_MEMORY_GB}GB, cleaning up...")
    
    # Upscale with timing
    start_time = time.time()
    try:
        if face_enhancer is not None:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=scale)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("⚠️  GPU out of memory, using tiled processing...")
            upsampler.tile = 400
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=scale)
        else:
            raise
    
    processing_time = time.time() - start_time
    
    # Validate output
    if output is None:
        raise ValueError("Upsampler returned None - model may have failed to process the image")
    
    # Save output with DPI metadata using PIL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure output is contiguous in memory (fixes tile grid issues)
    output = np.ascontiguousarray(output)
    
    # Apply film grain if requested
    if grain_strength > 0:
        output = add_film_grain(output, grain_strength)
        if progress_callback:
            progress_callback(f"Added film grain (strength: {grain_strength:.1f})")
            
    # Apply sharpening if requested
    if sharpen_amount > 0:
        output = apply_sharpening(output, sharpen_amount)
        if progress_callback:
            progress_callback(f"Applied sharpening (strength: {sharpen_amount:.1f})")
    
    # Convert BGR (OpenCV) to RGB (PIL)
    if len(output.shape) == 3 and output.shape[2] >= 3:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    else:
        output_rgb = output
    
    # Save with PIL to include DPI metadata
    pil_image = Image.fromarray(output_rgb)
    save_kwargs = {'dpi': (dpi, dpi)}
    
    # Preserve EXIF metadata from source image
    exif_data = None
    if preserve_metadata:
        try:
            with Image.open(str(input_path)) as src_img:
                exif_data = src_img.getexif()
                if exif_data:
                    save_kwargs['exif'] = exif_data
        except Exception as e:
            print(f"⚠️  Could not read EXIF from source: {e}")
    
    # Add format-specific options
    if output_format.lower() in ('jpg', 'jpeg'):
        save_kwargs['quality'] = 95
    elif output_format.lower() == 'png':
        save_kwargs['compress_level'] = 6
    
    pil_image.save(str(output_path), **save_kwargs)
    
    new_size = f"{output.shape[1]}x{output.shape[0]}"
    print(f"✅ Saved: {output_path.name} ({format_time(processing_time)})")
    print(f"   📐 {original_size} → {new_size}")
    
    return str(output_path)


def upscale_batch(
    input_dir: str,
    output_dir: str = None,
    model_name: str = "RealESRGAN_x4plus",
    scale: int = 4,
    face_enhance: bool = False,
    output_format: str = "png",
    sharpen_amount: float = 0.0
) -> list:
    """
    Upscale all images in a directory.
    
    Args:
        input_dir: Directory containing images
        output_dir: Output directory (auto-generated if None)
        model_name: Model to use
        scale: Upscaling factor
        face_enhance: Enable face enhancement
        output_format: Output format
    
    Returns:
        List of output paths
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "upscaled"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        return []
    
    print(f"📁 Found {len(images)} images to process")
    print(f"🧠 Memory limit: {MAX_MEMORY_GB}GB")
    
    results = []
    processing_times = []
    batch_start_time = time.time()
    
    for idx, img_path in enumerate(images):
        img_start_time = time.time()
        try:
            # Enforce memory limit before each image
            if not enforce_memory_limit():
                print(f"⚠️  Memory limit reached, waiting for cleanup...")
                time.sleep(1)  # Brief pause for garbage collection
            
            output_path = output_dir / f"{img_path.stem}.{output_format}"
            result = upscale_image(
                str(img_path),
                str(output_path),
                model_name,
                scale,
                face_enhance,
                output_format,
                sharpen_amount=sharpen_amount
            )
            results.append(result)
            
            # Track processing time for ETA calculation
            img_time = time.time() - img_start_time
            processing_times.append(img_time)
            
            # Calculate ETA
            avg_time = sum(processing_times) / len(processing_times)
            remaining_images = len(images) - (idx + 1)
            eta_seconds = avg_time * remaining_images
            
            # Progress display
            progress = (idx + 1) / len(images) * 100
            elapsed = time.time() - batch_start_time
            
            print(f"📊 Progress: {idx + 1}/{len(images)} ({progress:.1f}%) | "
                  f"Elapsed: {format_time(elapsed)} | "
                  f"ETA: {format_time(eta_seconds)} | "
                  f"Memory: {check_memory_usage():.1f}GB/{MAX_MEMORY_GB}GB")
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    total_time = time.time() - batch_start_time
    print(f"\n🎉 Completed! Processed {len(results)}/{len(images)} images in {format_time(total_time)}")
    if processing_times:
        print(f"   ⏱️  Average time per image: {format_time(sum(processing_times) / len(processing_times))}")
    print(f"   🧠 Peak memory usage: {check_memory_usage():.1f}GB")
    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="🖼️ Real-ESRGAN Image Upscaler - Free, Local AI Upscaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upscaler.py image.jpg                    # Upscale single image 4x
  python upscaler.py image.jpg -s 2               # Upscale 2x
  python upscaler.py ./photos -o ./output         # Batch process folder
  python upscaler.py portrait.jpg --face          # Enhance faces
  python upscaler.py anime.png -m anime           # Use anime model
        """
    )
    
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output path (file or directory)")
    parser.add_argument("-s", "--scale", type=int, choices=[2, 4], default=4,
                        help="Upscale factor (default: 4)")
    parser.add_argument("-m", "--model", choices=["general", "anime", "cartoon", "x2plus"], default="general",
                        help="Model type (default: general, cartoon for 3D characters)")
    parser.add_argument("--face", action="store_true",
                        help="Enable face enhancement (GFPGAN)")
    parser.add_argument("-f", "--format", choices=["png", "jpg", "webp"], default="png",
                        help="Output format (default: png)")
    parser.add_argument("--sharpen", type=float, default=0.0,
                        help="Sharpen amount (0.0 to 1.0, default: 0.0)")
    parser.add_argument("--download-models", action="store_true",
                        help="Download all models and exit")
    
    args = parser.parse_args()
    
    # Model name mapping
    model_map = {
        "general": "RealESRGAN_x4plus",
        "anime": "RealESRGAN_x4plus_anime_6B",
        "cartoon": "4x_foolhardy_Remacri",
        "realistic": "4x-UltraSharp",
        "crisp": "4x-UltraSharp",
        "portrait": "RealESRGAN_x4plus",
        "x2plus": "RealESRGAN_x2plus"
    }
    model_name = model_map[args.model]
    
    # Face enhancement is auto-enabled for portrait model in CLI too
    if args.model == "portrait":
        args.face = True
    
    # Handle scale for x2plus model
    if args.model == "x2plus":
        args.scale = 2
    
    # Download models if requested
    if args.download_models:
        print("📥 Downloading all models...")
        download_models()
        print("✅ All models downloaded!")
        return
    
    input_path = Path(args.input)
    
    print("=" * 50)
    print("🖼️  Real-ESRGAN Image Upscaler")
    print("=" * 50)
    print(f"   Model: {model_name}")
    print(f"   Scale: {args.scale}x")
    print(f"   Face Enhancement: {'Enabled' if args.face else 'Disabled'}")
    print(f"   Sharpening: {args.sharpen}")
    print("=" * 50)
    
    if input_path.is_dir():
        # Batch processing
        upscale_batch(
            str(input_path),
            args.output,
            model_name,
            args.scale,
            args.face,
            args.format,
            sharpen_amount=args.sharpen
        )
    elif input_path.is_file():
        # Single image
        upscale_image(
            str(input_path),
            args.output,
            model_name,
            args.scale,
            args.face,
            args.format,
            sharpen_amount=args.sharpen
        )
    else:
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
