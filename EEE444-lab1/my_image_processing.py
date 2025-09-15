# my_image_processing.py
# -*- coding: utf-8 -*-
"""
Συναρτήσεις επεξεργασίας εικόνας: MyProcess, MyProcess2, και βοηθητικά.
Απαιτεί: opencv-python, numpy, matplotlib (προαιρετικά για εμφάνιση).
"""
import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np


def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Μετατρέπει BGR (OpenCV) σε RGB (για οθόνη/Matplotlib)."""
    if img_bgr is None:
        raise ValueError("Αποτυχία φόρτωσης εικόνας (cv2.imread επέστρεψε None).")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _resize_keep_aspect(img: np.ndarray,
                        new_size: Union[int, Tuple[Optional[int], Optional[int]]]) -> np.ndarray:
    """
    Αλλαγή διαστάσεων με επιλογές:
    - int: κλιμακώνει τη ΜΙΚΡΗ διάσταση σε 'int' και υπολογίζει την άλλη (ίδιο aspect).
    - (w,h): μία από τις δύο μπορεί να είναι None => αυτόματος υπολογισμός (ίδιο aspect).
    - (w,h) πλήρη => άμεσο resize (ΠΡΟΣΟΧΗ: πιθανή παραμόρφωση αν δεν κρατήσετε ratio).
    """
    h, w = img.shape[:2]
    if isinstance(new_size, int):
        target_small = int(new_size)
        if w <= h:
            new_w = target_small
            new_h = int(round(h * (new_w / w)))
        else:
            new_h = target_small
            new_w = int(round(w * (new_h / h)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if isinstance(new_size, tuple):
        new_w, new_h = new_size
        if new_w is None and new_h is None:
            raise ValueError("Στο tuple new_size, τουλάχιστον μία διάσταση πρέπει να δοθεί.")
        if new_w is None:
            # υπολογίζω πλάτος με ίδιο ratio
            scale = new_h / h
            new_w = int(round(w * scale))
        elif new_h is None:
            scale = new_w / w
            new_h = int(round(h * scale))
        else:
            new_w, new_h = int(new_w), int(new_h)
        return cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)

    raise TypeError("new_size πρέπει να είναι int ή tuple(width, height) με None αποδεκτό.")


def _rotate_no_crop(img: np.ndarray, angle_deg: float,
                    border_mode=cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Περιστροφή χωρίς αποκοπή: επεκτείνει τον καμβά ώστε τίποτα να μη χαθεί.
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # υπολογισμός νέου μεγέθους
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # προσαρμογή μετάθεσης για να κεντράρει
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=border_mode,
                             borderValue=(0, 0, 0))
    return rotated


def _negative(img: np.ndarray) -> np.ndarray:
    """Αρνητική εικόνα (bitwise not)."""
    return cv2.bitwise_not(img)


def MyProcess(image_path: str,
              new_size: Union[int, Tuple[Optional[int], Optional[int]]],
              angle_deg: float,
              show: bool = True,
              out_path: Optional[str] = None) -> np.ndarray:
    """
    Φορτώνει -> resize (με ratio όταν ζητηθεί) -> περιστροφή χωρίς αποκοπή -> αρνητικό.
    Επιστρέφει την τελική RGB εικόνα.
    Parameters
    ----------
    image_path : str
        Διαδρομή εικόνας.
    new_size : int ή (w,h) με δυνατότητα None.
    angle_deg : float
        Γωνία σε μοίρες (θετική: αριστερόστροφα).
    show : bool
        Αν True, εμφανίζει σε παράθυρο (matplotlib ή cv2.imshow).
    out_path : Optional[str]
        Αν δοθεί, αποθηκεύει (σε BGR για OpenCV).
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = _ensure_rgb(bgr)
    img = _resize_keep_aspect(img, new_size)
    img = _rotate_no_crop(img, angle_deg)
    img = _negative(img)

    if out_path:
        # Αποθήκευση σε BGR
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if show:
        # Εμφάνιση με OpenCV (BGR) ή με matplotlib (RGB).
        cv2.imshow("Result (RGB shown as BGR window)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def MyProcess2(**kwargs) -> np.ndarray:
    """
    Ευέλικτη συνάρτηση τύπου 'varargin' (keyword args).
    Προτεραιότητα: image_path > subject_index.
    Keyword Args
    ------------
    subject_index : int
        Φορτώνει images_dir/subject<index>.jpg αν δεν έχει δοθεί image_path.
    image_path : str
        Ρητή διαδρομή εικόνας (έχει προτεραιότητα).
    resize : int ή (w,h) με None αποδεκτό
        Αλλαγή διαστάσεων με διατήρηση αναλογιών όταν μία διάσταση είναι None
        ή όταν δίνεται int (μικρή διάσταση).
    rotate : float
        Γωνία περιστροφής (μοίρες).
    negative : bool
        Εφαρμογή αρνητικού.
    show : bool
        Εμφάνιση αποτελέσματος.
    out_path : str
        Αποθήκευση αποτελέσματος.
    images_dir : str
        Φάκελος για subject<x>.jpg (default: "./images").
    Επιστρέφει: RGB εικόνα (np.ndarray).
    """
    images_dir = kwargs.get("images_dir", "./images")
    subject_index = kwargs.get("subject_index", None)
    image_path = kwargs.get("image_path", None)
    resize = kwargs.get("resize", None)
    rotate = kwargs.get("rotate", None)
    negative = kwargs.get("negative", None)
    show = kwargs.get("show", True)
    out_path = kwargs.get("out_path", None)

    # Εύρεση εικόνας
    if image_path is None:
        if subject_index is None:
            # default fallback
            image_path = os.path.join(images_dir, "subject1.jpg")
        else:
            image_path = os.path.join(images_dir, f"subject{int(subject_index)}.jpg")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = _ensure_rgb(bgr)

    # Μετασχηματισμοί κατά επιλογή
    if resize is not None:
        img = _resize_keep_aspect(img, resize)
    if rotate is not None:
        img = _rotate_no_crop(img, float(rotate))
    if negative is True:
        img = _negative(img)

    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if show:
        cv2.imshow("MyProcess2 Result (RGB shown as BGR window)", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# --------- Συμπληρωματική: Ιστόγραμμα ---------
def histogram_manual(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Χειροκίνητο ιστόγραμμα [0..255] με NumPy.
    Δέχεται gray uint8. Επιστρέφει μέτρηση ανά επίπεδο.
    """
    if gray.dtype != np.uint8:
        raise TypeError("Το histogram_manual αναμένει uint8 gray image.")
    # np.bincount με fixed length 256
    hist = np.bincount(gray.ravel(), minlength=bins)
    return hist


def histogram_opencv(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Ιστόγραμμα με OpenCV: cv2.calcHist.
    """
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
    return hist


def load_to_gray(path: str) -> np.ndarray:
    """Φορτώνει εικόνα και επιστρέφει γκρίζα (uint8)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"Αδύνατη φόρτωση: {path}")
    return g
