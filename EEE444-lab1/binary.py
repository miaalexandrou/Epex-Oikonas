import cv2
import numpy as np

def convert_to_binary(img_rgb, threshold=127):
    """Convert RGB image to binary using a given threshold."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def convert_to_grayscale(img_rgb):
    """Convert RGB image to grayscale and return as RGB format."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def process_binary_operations(img_rgb, operation="binary", threshold=127):
    """
    Process image with binary operations.
    
    Args:
        img_rgb: Input RGB image as numpy array
        operation: Type of operation - "binary", "grayscale", or "both"
        threshold: Threshold value for binary conversion (0-255)
    
    Returns:
        Processed image as RGB numpy array
    """
    if operation == "grayscale":
        return convert_to_grayscale(img_rgb)
    elif operation == "binary":
        return convert_to_binary(img_rgb, threshold)
    elif operation == "both":
        # First convert to grayscale, then to binary
        gray_rgb = convert_to_grayscale(img_rgb)
        return convert_to_binary(gray_rgb, threshold)
    else:
        # Default to binary if unknown operation
        return convert_to_binary(img_rgb, threshold)