import cv2
import numpy as np

def get_kernel(shape: str, size: int):
    if size % 2 == 0:
        size += 1
        
    if shape == 'square':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == 'diamond':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

def apply_threshold(image: np.ndarray, threshold: int):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def apply_operation(image: np.ndarray, operation: str, kernel: np.ndarray):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    if operation == 'not':
        result = cv2.bitwise_not(gray)
    elif operation == 'and':
        # Apply erosion followed by dilation (like opening) for AND-like behavior
        result = cv2.erode(gray, kernel, iterations=1)
    elif operation == 'or':
        # Apply dilation followed by erosion (like closing) for OR-like behavior  
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'xor':
        shifted = np.roll(gray, 5, axis=1)
        result = cv2.bitwise_xor(gray, shifted)
    elif operation == 'dilate':
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation == 'erode':
        result = cv2.erode(gray, kernel, iterations=1)
    elif operation == 'open':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    else:
        result = gray
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def on_apply_morphology_gui(main_window):
    try:
        if main_window._before_rgb is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        threshold = main_window.morphThresholdSlider.value()
        kernel_shape = main_window.morphKernelShape.currentText()
        kernel_size = int(main_window.morphKernelSize.currentText())
        operation = main_window.morphOperation.currentText()
        
        thresh_image = apply_threshold(main_window._before_rgb, threshold)
        kernel = get_kernel(kernel_shape, kernel_size)
        result_rgb = apply_operation(thresh_image, operation, kernel)
        
        main_window._after_rgb = result_rgb
        
        from app_gui import np_rgb_to_qpixmap
        pix = np_rgb_to_qpixmap(result_rgb, main_window.lblAfter.size())
        main_window.zoomHandlerAfter.set_pixmap(pix)
        main_window.centerTabs.setCurrentWidget(main_window.tabProcessed)
        main_window.rightTabs.setCurrentIndex(2)
        
    except ValueError as e:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.information(main_window, "Info", str(e))
    except Exception as e:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.critical(main_window, "Σφάλμα", str(e))