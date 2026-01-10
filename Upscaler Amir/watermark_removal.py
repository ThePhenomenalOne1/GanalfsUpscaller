"""
Watermark Removal Module
Detects and removes AI-generated watermarks (like Gemini sparkle) using template matching and inpainting.
"""

import cv2
import numpy as np
from pathlib import Path


# Gemini sparkle watermark template - encoded as a simple pattern
# The watermark is typically a 4-pointed star in the bottom-right corner
# We'll create an approximate template and use flexible matching

def create_gemini_template():
    """
    Create an approximate template for the Gemini sparkle watermark.
    The sparkle is a 4-pointed star shape, typically semi-transparent.
    """
    # Create a small template image (the sparkle is roughly 40-60 pixels)
    size = 48
    template = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    # Draw a 4-pointed star shape
    # Vertical line
    cv2.line(template, (center, 2), (center, size-3), 255, 2)
    # Horizontal line  
    cv2.line(template, (2, center), (size-3, center), 255, 2)
    # Diagonal lines (shorter)
    offset = size // 4
    cv2.line(template, (center-offset, center-offset), (center+offset, center+offset), 200, 1)
    cv2.line(template, (center+offset, center-offset), (center-offset, center+offset), 200, 1)
    
    # Add a slight blur to match the watermark's appearance
    template = cv2.GaussianBlur(template, (3, 3), 0)
    
    return template


def detect_watermark(image, search_region_ratio=0.25):
    """
    Detect if a Gemini-style watermark exists in the bottom-right corner.
    
    Args:
        image: BGR image (numpy array)
        search_region_ratio: How much of the image to search (0.25 = bottom-right 25%)
    
    Returns:
        tuple: (found, x, y, w, h) - whether found and bounding box if found
    """
    h, w = image.shape[:2]
    
    # Define search region (bottom-right corner)
    search_h = int(h * search_region_ratio)
    search_w = int(w * search_region_ratio)
    roi = image[h-search_h:h, w-search_w:w]
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Method 1: Look for bright spots in the corner (watermarks are usually light)
    # Apply threshold to find bright areas
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright spots
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for star-like shapes (roughly square aspect ratio, small size)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Watermark characteristics:
        # - Roughly square (aspect ratio close to 1)
        # - Small relative to image (20-80 pixels typically)
        # - Located in corner region
        aspect_ratio = cw / max(ch, 1)
        
        if 0.5 < aspect_ratio < 2.0 and 100 < area < 5000 and 15 < cw < 100:
            # Likely a watermark - expand the bounding box slightly
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            cw = min(search_w - x, cw + padding * 2)
            ch = min(search_h - y, ch + padding * 2)
            
            # Convert coordinates to full image space
            full_x = w - search_w + x
            full_y = h - search_h + y
            
            return True, full_x, full_y, cw, ch
    
    # Method 2: Template matching as fallback
    template = create_gemini_template()
    
    # Try multiple scales
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
            continue
            
        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.4:  # Good match threshold
            tw, th = scaled_template.shape[::-1]
            # Convert to full image coordinates
            full_x = w - search_w + max_loc[0]
            full_y = h - search_h + max_loc[1]
            
            # Add padding for inpainting
            padding = 10
            return True, max(0, full_x - padding), max(0, full_y - padding), tw + padding*2, th + padding*2
    
    return False, 0, 0, 0, 0


def remove_watermark(image, x, y, w, h, method='inpaint'):
    """
    Remove watermark from image using inpainting.
    
    Args:
        image: BGR image (numpy array)
        x, y, w, h: Bounding box of watermark
        method: 'inpaint' for OpenCV inpainting, 'blur' for simple blur
    
    Returns:
        Processed image with watermark removed
    """
    result = image.copy()
    img_h, img_w = image.shape[:2]
    
    # Ensure bounds are valid
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return result
    
    if method == 'inpaint':
        # Create mask for inpainting
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # Create an elliptical mask for smoother blending
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.ellipse(mask, (center_x, center_y), (w // 2, h // 2), 0, 0, 360, 255, -1)
        
        # Use Navier-Stokes based inpainting (better for textures)
        result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        
    elif method == 'blur':
        # Simple blur replacement
        roi = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)
        result[y:y+h, x:x+w] = blurred
    
    return result


def process_watermark_removal(image, enabled=True):
    """
    Main function to detect and remove watermark from an image.
    
    Args:
        image: BGR image (numpy array)
        enabled: Whether watermark removal is enabled
    
    Returns:
        tuple: (processed_image, watermark_found)
    """
    if not enabled:
        return image, False
    
    # Detect watermark
    found, x, y, w, h = detect_watermark(image)
    
    if found:
        # Remove the watermark
        result = remove_watermark(image, x, y, w, h, method='inpaint')
        return result, True
    
    return image, False


# Alternative: Simple corner crop method (more reliable but loses some image)
def crop_watermark_corner(image, crop_size=60):
    """
    Simply crop the bottom-right corner where watermarks typically appear.
    This is more reliable but loses a small portion of the image.
    
    Args:
        image: BGR image
        crop_size: Size of corner to crop (pixels)
    
    Returns:
        Image with corner cropped and filled with nearby content
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Scale crop size based on image size
    scaled_crop = max(30, min(crop_size, min(h, w) // 10))
    
    # Create mask for the corner
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Triangle/rounded corner in bottom-right
    pts = np.array([
        [w, h],
        [w - scaled_crop * 2, h],
        [w, h - scaled_crop * 2]
    ], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Inpaint the corner
    result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    return result
