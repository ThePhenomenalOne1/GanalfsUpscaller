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


def download_models():
    """Download required models if not present."""
    import urllib.request
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        # Specialized cartoon/illustration model from OpenModelDB - highly rated for 3D cartoon upscaling
        "4x_foolhardy_Remacri.pth": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    }
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"📥 Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ Downloaded {model_name}")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
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
        print("Models not found. Downloading...")
        download_models()
    
    if progress_callback:
        progress_callback("Step 2/5: Configuring model architecture...")
    
    # Configure model architecture based on model type
    if "anime" in model_name.lower():
        # Anime model has different architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif "x2plus" in model_name.lower():
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
    elif "remacri" in model_name.lower() or "foolhardy" in model_name.lower():
        # 4x_foolhardy_Remacri - specialized cartoon model (RRDB 23 blocks, 4x scale)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        # Default x4plus model
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
    
    # For community models (like Remacri), we need to load weights manually
    # because they use a different state dict format
    if "remacri" in model_name.lower() or "foolhardy" in model_name.lower():
        import torch
        
        if progress_callback:
            progress_callback("Loading community model (Remacri)...")
        
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
            print(f"Strict loading failed: {e}")
            print("Trying non-strict loading...")
            
            # Try non-strict loading
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded model with strict=False (some keys may be missing)")
            except RuntimeError as e2:
                print(f"Non-strict loading also failed: {e2}")
                
                # Try key remapping for ESRGAN -> Real-ESRGAN format
                print("Trying key remapping...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Common remappings
                    new_key = k
                    # ESRGAN format: RRDB_trunk.X.rdb... -> body.X.rdb...
                    if k.startswith('RRDB_trunk.'):
                        new_key = k.replace('RRDB_trunk.', 'body.')
                    # trunk_conv -> conv_body
                    if k.startswith('trunk_conv.'):
                        new_key = k.replace('trunk_conv.', 'conv_body.')
                    new_state_dict[new_key] = v
                
                try:
                    model.load_state_dict(new_state_dict, strict=False)
                    print("Loaded model with key remapping")
                except Exception as e3:
                    raise RuntimeError(f"Failed to load Remacri model: {e3}")
        
        model.eval()
        
        if gpu_available:
            model = model.to(torch.device('cuda'))
        
        # Create a minimal RealESRGANer without loading (we already loaded)
        # We need to monkey-patch to avoid the internal load
        class PreloadedUpsampler:
            def __init__(self, model, scale, tile, tile_pad, pre_pad, half, device):
                self.model = model
                self.scale = scale
                self.tile = tile
                self.tile_pad = tile_pad
                self.pre_pad = pre_pad
                self.half = half
                self.device = device
                if self.half:
                    self.model = self.model.half()
            
            def enhance(self, img, outscale=None):
                import torch
                import numpy as np
                
                if outscale is None:
                    outscale = self.scale
                
                # Convert to tensor
                img = img.astype(np.float32) / 255.0
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                
                img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
                img = img.unsqueeze(0).to(self.device)
                
                if self.half:
                    img = img.half()
                
                # Process with tiling if needed
                with torch.no_grad():
                    if self.tile > 0:
                        output = self._tile_process(img)
                    else:
                        output = self.model(img)
                
                # Convert back
                output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output, (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                
                return output, None
            
            def _tile_process(self, img):
                import torch
                batch, channel, height, width = img.shape
                output_height = height * self.scale
                output_width = width * self.scale
                output = img.new_zeros((batch, channel, output_height, output_width))
                
                tiles_x = (width + self.tile - 1) // self.tile
                tiles_y = (height + self.tile - 1) // self.tile
                
                for y in range(tiles_y):
                    for x in range(tiles_x):
                        tile_idx = y * tiles_x + x + 1
                        print(f"        Tile {tile_idx}/{tiles_x * tiles_y}")
                        
                        ofs_x = x * self.tile
                        ofs_y = y * self.tile
                        
                        input_start_x = ofs_x
                        input_end_x = min(ofs_x + self.tile, width)
                        input_start_y = ofs_y
                        input_end_y = min(ofs_y + self.tile, height)
                        
                        input_tile = img[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        
                        with torch.no_grad():
                            output_tile = self.model(input_tile)
                        
                        output_start_x = input_start_x * self.scale
                        output_end_x = input_end_x * self.scale
                        output_start_y = input_start_y * self.scale
                        output_end_y = input_end_y * self.scale
                        
                        output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile
                
                return output
        
        device = torch.device('cuda' if gpu_available else 'cpu')
        upsampler = PreloadedUpsampler(model, scale, tile_size, 10, 0, half_precision and gpu_available, device)
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
    remove_watermark: bool = False
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
        output_path = output_dir / f"{input_path.stem}{suffix}_{scale}x.{output_format}"
    
    output_path = Path(output_path)
    
    print(f"🖼️  Processing: {input_path.name}")
    
    # Read image
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")
    
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
    output_format: str = "png"
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
            
            output_path = output_dir / f"{img_path.stem}_upscaled_{scale}x.{output_format}"
            result = upscale_image(
                str(img_path),
                str(output_path),
                model_name,
                scale,
                face_enhance,
                output_format
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
    parser.add_argument("--download-models", action="store_true",
                        help="Download all models and exit")
    
    args = parser.parse_args()
    
    # Model name mapping
    model_map = {
        "general": "RealESRGAN_x4plus",
        "anime": "RealESRGAN_x4plus_anime_6B",
        "cartoon": "4x_foolhardy_Remacri",  # Specialized model for 3D cartoon characters
        "x2plus": "RealESRGAN_x2plus"
    }
    model_name = model_map[args.model]
    
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
    print("=" * 50)
    
    if input_path.is_dir():
        # Batch processing
        upscale_batch(
            str(input_path),
            args.output,
            model_name,
            args.scale,
            args.face,
            args.format
        )
    elif input_path.is_file():
        # Single image
        upscale_image(
            str(input_path),
            args.output,
            model_name,
            args.scale,
            args.face,
            args.format
        )
    else:
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
