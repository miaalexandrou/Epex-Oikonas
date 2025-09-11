import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES_DIR = os.path.join(APP_DIR, "images")

def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("Αποτυχία φόρτωσης εικόνας.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _resize_keep_aspect(img, new_size):
    h, w = img.shape[:2]
    if isinstance(new_size, int):
        if w <= h:
            new_w = new_size
            new_h = int(h * new_w / w)
        else:
            new_h = new_size
            new_w = int(w * new_h / h)
        return cv2.resize(img, (new_w, new_h))
    if isinstance(new_size, tuple):
        new_w, new_h = new_size
        if new_w is None:
            new_w = int(w * (new_h / h))
        if new_h is None:
            new_h = int(h * (new_w / w))
        return cv2.resize(img, (new_w, new_h))
    return img

def _rotate_no_crop(img, angle_deg):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    new_w = int(h*sin + w*cos)
    new_h = int(h*cos + w*sin)
    M[0,2] += (new_w/2) - (w/2)
    M[1,2] += (new_h/2) - (h/2)
    return cv2.warpAffine(img, M, (new_w, new_h))

def _negative(img): 
    return cv2.bitwise_not(img)

def _binary(img, threshold=127):
    # Convert to grayscale if the image is in color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Convert back to RGB to maintain compatibility with other functions
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def MyProcess(image_path, new_size, angle_deg, show=True, out_path=None):
    bgr = cv2.imread(image_path)
    img = _ensure_rgb(bgr)
    img = _resize_keep_aspect(img, new_size)
    img = _rotate_no_crop(img, angle_deg)
    img = _negative(img)
    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if show:
        cv2.imshow("Result", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

def MyProcess2(**kwargs):
    image_path = kwargs.get("image_path")
    subject_index = kwargs.get("subject_index")
    images_dir = kwargs.get("images_dir", DEFAULT_IMAGES_DIR)
    resize = kwargs.get("resize")
    rotate = kwargs.get("rotate")
    negative = kwargs.get("negative", False)
    show = kwargs.get("show", True)
    out_path = kwargs.get("out_path")

    if image_path is None and subject_index is not None:
        image_path = os.path.join(images_dir, f"subject{subject_index}.jpg")
    
    bgr = cv2.imread(image_path)
    img = _ensure_rgb(bgr)
    if resize: 
        img = _resize_keep_aspect(img, resize)
    if rotate: 
        img = _rotate_no_crop(img, rotate)
    if negative: 
        img = _negative(img)
    
    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if show:
        cv2.imshow("Result", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img