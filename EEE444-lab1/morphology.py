import cv2
import numpy as np

class MorphologyProcessor:
    """Class for handling morphological operations on images."""
    
    @staticmethod
    def get_kernel(shape: str, size: int):
        """
        Create a kernel for morphological operations.
        
        Args:
            shape: Shape of the kernel ('square', 'diamond', 'cross')
            size: Size of the kernel (odd number)
        
        Returns:
            numpy.ndarray: Kernel for morphological operations
        """
        if size % 2 == 0:
            size += 1  # Ensure odd size
            
        if shape == 'square':
            return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        elif shape == 'diamond':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        elif shape == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
        else:
            # Default to square
            return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    
    @staticmethod
    def apply_threshold(image: np.ndarray, threshold: int, threshold_type: str = 'binary'):
        """
        Apply thresholding to an image.
        
        Args:
            image: Input image (RGB)
            threshold: Threshold value (0-255)
            threshold_type: Type of thresholding ('binary', 'binary_inv', 'adaptive_mean', 'adaptive_gaussian')
        
        Returns:
            numpy.ndarray: Thresholded image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if threshold_type == 'binary':
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        elif threshold_type == 'binary_inv':
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == 'adaptive_mean':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == 'adaptive_gaussian':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert back to RGB for display
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def apply_morphological_operation(image: np.ndarray, operation: str, kernel: np.ndarray):
        """
        Apply morphological operation to an image.
        
        Args:
            image: Input image (should be binary/grayscale)
            operation: Type of operation ('not', 'and', 'or', 'xor', 'dilate', 'erode', 'open', 'close')
            kernel: Morphological kernel
        
        Returns:
            numpy.ndarray: Processed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if operation == 'not':
            result = cv2.bitwise_not(gray)
        elif operation == 'and':
            # For demonstration, AND with itself (could be modified to AND with another image)
            result = cv2.bitwise_and(gray, gray)
        elif operation == 'or':
            # For demonstration, OR with itself (could be modified to OR with another image)
            result = cv2.bitwise_or(gray, gray)
        elif operation == 'xor':
            # For demonstration, XOR with a shifted version of itself
            shifted = np.roll(gray, 5, axis=1)  # Shift 5 pixels right
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
            result = gray  # No operation
        
        # Convert back to RGB for display
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def process_morphology(image: np.ndarray, threshold: int = 127, threshold_type: str = 'binary',
                          kernel_shape: str = 'rectangle', kernel_size: int = 5, operation: str = 'erosion'):
        """
        Complete morphological processing pipeline.
        
        Args:
            image: Input RGB image
            threshold: Threshold value for binarization
            threshold_type: Type of thresholding
            kernel_shape: Shape of morphological kernel
            kernel_size: Size of morphological kernel
            operation: Morphological operation to apply
        
        Returns:
            numpy.ndarray: Processed RGB image
        """
        # Step 1: Apply thresholding
        thresh_image = MorphologyProcessor.apply_threshold(image, threshold, threshold_type)
        
        # Step 2: Create kernel
        kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        # Step 3: Apply morphological operation
        result = MorphologyProcessor.apply_morphological_operation(thresh_image, operation, kernel)
        
        return result

def process_morphology_simple(image: np.ndarray, threshold: int = 127, kernel_shape: str = 'rectangle', 
                             kernel_size: int = 5, operation: str = 'erosion'):
    """
    Simple function for morphological processing (for backward compatibility).
    
    Args:
        image: Input RGB image
        threshold: Threshold value
        kernel_shape: Kernel shape
        kernel_size: Kernel size
        operation: Morphological operation
    
    Returns:
        numpy.ndarray: Processed RGB image
    """
    return MorphologyProcessor.process_morphology(
        image=image,
        threshold=threshold,
        threshold_type='binary',
        kernel_shape=kernel_shape,
        kernel_size=kernel_size,
        operation=operation
    )

def apply_morphology_to_gui(before_rgb, morph_threshold_slider, morph_kernel_shape, 
                           morph_kernel_size, morph_operation):
    """
    Apply morphological operations for GUI integration.
    
    Args:
        before_rgb: Input RGB image
        morph_threshold_slider: Threshold slider widget (has .value() method)
        morph_kernel_shape: Kernel shape combobox (has .currentText() method)
        morph_kernel_size: Kernel size combobox (has .currentText() method)
        morph_operation: Operation combobox (has .currentText() method)
    
    Returns:
        numpy.ndarray: Processed RGB image
    
    Raises:
        Exception: If processing fails
    """
    if before_rgb is None:
        raise ValueError("No image loaded. Please load an image first.")
    
    try:
        # Get parameters from GUI controls
        threshold = morph_threshold_slider.value()
        kernel_shape = morph_kernel_shape.currentText()
        kernel_size = int(morph_kernel_size.currentText())
        operation = morph_operation.currentText()
        
        # Apply morphological processing
        result_rgb = process_morphology_simple(
            image=before_rgb,
            threshold=threshold,
            kernel_shape=kernel_shape,
            kernel_size=kernel_size,
            operation=operation
        )
        
        return result_rgb
        
    except Exception as e:
        raise Exception(f"Error in morphological processing: {str(e)}")

def on_apply_morphology_gui(main_window):
    """
    Complete morphology application function that handles GUI updates.
    This function replaces the on_apply_morphology method in the MainWindow class.
    
    Args:
        main_window: Reference to the MainWindow instance with all GUI components
    """
    try:
        # Use the morphology module function
        result_rgb = apply_morphology_to_gui(
            main_window._before_rgb,
            main_window.morphThresholdSlider,
            main_window.morphKernelShape,
            main_window.morphKernelSize,
            main_window.morphOperation
        )
        
        # Update the result
        main_window._after_rgb = result_rgb
        
        # Import the helper function for pixmap conversion
        from app_gui import np_rgb_to_qpixmap
        pix = np_rgb_to_qpixmap(result_rgb, main_window.lblAfter.size())
        main_window.zoomHandlerAfter.set_pixmap(pix)
        main_window.centerTabs.setCurrentWidget(main_window.tabProcessed)
        
        # Switch to Morphology tab to show the controls were applied
        main_window.rightTabs.setCurrentIndex(2)  # Morphology tab is index 2
        
    except ValueError as e:
        # Import QtWidgets for message boxes
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.information(main_window, "Info", str(e))
    except Exception as e:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.critical(main_window, "Σφάλμα", str(e))