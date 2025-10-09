import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------- Fourier Processing Functions --------------------------

def freq_grid(shape):
    """Generate frequency distance grid centered at origin."""
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U**2 + V**2)

def gaussian_lowpass(shape, D0):
    """Create Gaussian lowpass filter."""
    D = freq_grid(shape)
    return np.exp(-(D**2)/(2*(D0**2 + 1e-8))).astype(np.float32)

def gaussian_highpass(shape, D0):
    """Create Gaussian highpass filter."""
    return 1.0 - gaussian_lowpass(shape, D0)

def butterworth_lowpass(shape, D0, n):
    """Create Butterworth lowpass filter."""
    D = freq_grid(shape)
    H = 1 / (1 + (D / (D0 + 1e-8))**(2*n))
    return H.astype(np.float32)

def butterworth_highpass(shape, D0, n):
    """Create Butterworth highpass filter."""
    return 1.0 - butterworth_lowpass(shape, D0, n)

def ideal_lowpass(shape, D0):
    """Create ideal lowpass filter."""
    D = freq_grid(shape)
    H = (D <= D0).astype(np.float32)
    return H

def ideal_highpass(shape, D0):
    """Create ideal highpass filter."""
    return 1.0 - ideal_lowpass(shape, D0)

def bandpass_filter(shape, D_low, D_high, kind="gaussian", n=2):
    """Create bandpass filter."""
    if D_low >= D_high:
        raise ValueError("D_low must be less than D_high")
    
    if kind == "gaussian":
        H_low = gaussian_highpass(shape, D_low)
        H_high = gaussian_lowpass(shape, D_high)
    elif kind == "butterworth":
        H_low = butterworth_highpass(shape, D_low, n)
        H_high = butterworth_lowpass(shape, D_high, n)
    elif kind == "ideal":
        H_low = ideal_highpass(shape, D_low)
        H_high = ideal_lowpass(shape, D_high)
    else:
        raise ValueError("Unknown filter kind")
    
    return H_low * H_high

def notch_reject_filter(shape, centers, radius):
    """Create notch reject filter."""
    H = np.ones(shape, dtype=np.float32)
    rows, cols = shape
    center_u, center_v = rows // 2, cols // 2
    
    for center_str in centers:
        try:
            # Parse center coordinates
            center_str = center_str.strip("() ")
            u_offset, v_offset = map(int, center_str.split(","))
            
            # Create notch at (u_offset, v_offset) from center
            u_pos = center_u + u_offset
            v_pos = center_v + v_offset
            
            # Create circular mask
            u_grid, v_grid = np.ogrid[:rows, :cols]
            mask1 = (u_grid - u_pos)**2 + (v_grid - v_pos)**2 <= radius**2
            H[mask1] = 0
            
            # Also create symmetric notch
            u_sym = center_u - u_offset
            v_sym = center_v - v_offset
            mask2 = (u_grid - u_sym)**2 + (v_grid - v_sym)**2 <= radius**2
            H[mask2] = 0
            
        except Exception as e:
            print(f"Error parsing notch center '{center_str}': {e}")
    
    return H

def apply_filter_gray(img_rgb, H_shift):
    """Apply frequency domain filter to grayscale image."""
    # Convert to grayscale
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    
    gray = gray.astype(np.float32)
    
    # FFT and shift
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    
    # Apply filter
    F_filtered = F_shift * H_shift
    
    # IFFT
    F_ishifted = np.fft.ifftshift(F_filtered)
    result = np.fft.ifft2(F_ishifted)
    result = np.real(result)
    
    # Normalize to uint8
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def apply_filter_rgb(img_rgb, H_shift):
    """Apply frequency domain filter to RGB image."""
    result = np.zeros_like(img_rgb, dtype=np.uint8)
    
    for i in range(3):  # RGB channels
        channel = img_rgb[:, :, i].astype(np.float32)
        
        # FFT and shift
        F = np.fft.fft2(channel)
        F_shift = np.fft.fftshift(F)
        
        # Apply filter
        F_filtered = F_shift * H_shift
        
        # IFFT
        F_ishifted = np.fft.ifftshift(F_filtered)
        result_channel = np.fft.ifft2(F_ishifted)
        result_channel = np.real(result_channel)
        
        # Normalize to uint8
        result_channel = cv2.normalize(result_channel, None, 0, 255, cv2.NORM_MINMAX)
        result[:, :, i] = result_channel.astype(np.uint8)
    
    return result

def compute_spectrum(img_rgb):
    """Compute and return the magnitude spectrum for display."""
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    
    gray = gray.astype(np.float32)
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    
    # Compute log magnitude spectrum
    magnitude = np.abs(F_shift)
    spectrum = np.log(1 + magnitude)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    
    return spectrum.astype(np.uint8)

def phase_only_reconstruction(img_rgb):
    """Reconstruct image using phase information only."""
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    
    gray = gray.astype(np.float32)
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    
    # Keep phase, set magnitude to 1
    phase = np.angle(F_shift)
    F_phase_only = np.exp(1j * phase)
    
    # IFFT
    F_ishifted = np.fft.ifftshift(F_phase_only)
    result = np.fft.ifft2(F_ishifted)
    result = np.real(result)
    
    # Normalize to uint8
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def magnitude_only_reconstruction(img_rgb):
    """Reconstruct image using magnitude information only."""
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
    
    gray = gray.astype(np.float32)
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    
    # Keep magnitude, set phase to 0
    magnitude = np.abs(F_shift)
    F_mag_only = magnitude  # Phase = 0
    
    # IFFT
    F_ishifted = np.fft.ifftshift(F_mag_only)
    result = np.fft.ifft2(F_ishifted)
    result = np.real(result)
    
    # Normalize to uint8
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

# -------------------------- Image Canvas for Results --------------------------

class ImageCanvas(QtWidgets.QLabel):
    """Simple image display widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("border: 1px dashed #3a3c42;")
        self.setText("— no result —")
        self._pixmap = None
    
    def set_image_rgb(self, img_rgb):
        """Set image from RGB numpy array."""
        if img_rgb.ndim == 2:
            # Convert grayscale to RGB for display
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # Scale to fit widget
        scaled_pixmap = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self._pixmap = pixmap
    
    def resizeEvent(self, event):
        """Rescale pixmap on resize."""
        super().resizeEvent(event)
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

# -------------------------- GUI Functions --------------------------

def create_fourier_tab(parent):
    """Create and return the Fourier tab widget."""
    tab_fourier = QtWidgets.QWidget()
    
    # Main layout
    layout = QtWidgets.QVBoxLayout(tab_fourier)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)
    
    # Parameters group
    grp_params = QtWidgets.QGroupBox("Parameters")
    grid = QtWidgets.QGridLayout(grp_params)
    
    # Filter kind
    lbl_kind = QtWidgets.QLabel("Filter Kind:")
    cmb_kind = QtWidgets.QComboBox()
    cmb_kind.addItems(["gaussian", "butterworth", "ideal"])
    cmb_kind.setCurrentText("gaussian")
    grid.addWidget(lbl_kind, 0, 0)
    grid.addWidget(cmb_kind, 0, 1)
    
    # D0 (cutoff frequency)
    lbl_d0 = QtWidgets.QLabel("D0 (cutoff):")
    spn_d0 = QtWidgets.QSpinBox()
    spn_d0.setRange(1, 500)
    spn_d0.setValue(30)
    grid.addWidget(lbl_d0, 0, 2)
    grid.addWidget(spn_d0, 0, 3)
    
    # Order n (for Butterworth)
    lbl_n = QtWidgets.QLabel("Order n:")
    spn_n = QtWidgets.QSpinBox()
    spn_n.setRange(1, 10)
    spn_n.setValue(2)
    grid.addWidget(lbl_n, 1, 0)
    grid.addWidget(spn_n, 1, 1)
    
    # Band low
    lbl_d_low = QtWidgets.QLabel("D_low:")
    spn_d_low = QtWidgets.QSpinBox()
    spn_d_low.setRange(1, 500)
    spn_d_low.setValue(20)
    grid.addWidget(lbl_d_low, 1, 2)
    grid.addWidget(spn_d_low, 1, 3)
    
    # Band high
    lbl_d_high = QtWidgets.QLabel("D_high:")
    spn_d_high = QtWidgets.QSpinBox()
    spn_d_high.setRange(1, 500)
    spn_d_high.setValue(50)
    grid.addWidget(lbl_d_high, 2, 0)
    grid.addWidget(spn_d_high, 2, 1)
    
    # Notch centers
    lbl_notch = QtWidgets.QLabel("Notch centers:")
    edt_notch = QtWidgets.QLineEdit()
    edt_notch.setPlaceholderText("e.g. (25,0);(-25,0)")
    edt_notch.setText("(25,0);(-25,0)")
    grid.addWidget(lbl_notch, 2, 2)
    grid.addWidget(edt_notch, 2, 3)
    
    # Notch radius
    lbl_radius = QtWidgets.QLabel("Notch radius:")
    spn_radius = QtWidgets.QSpinBox()
    spn_radius.setRange(1, 50)
    spn_radius.setValue(5)
    grid.addWidget(lbl_radius, 3, 0)
    grid.addWidget(spn_radius, 3, 1)
    
    # Apply on RGB
    chk_rgb = QtWidgets.QCheckBox("Apply on RGB")
    chk_rgb.setChecked(False)
    grid.addWidget(chk_rgb, 3, 2)
    
    layout.addWidget(grp_params)
    
    # Actions group
    grp_actions = QtWidgets.QGroupBox("Actions")
    h_actions = QtWidgets.QHBoxLayout(grp_actions)
    
    btn_spectrum = QtWidgets.QPushButton("Show Spectrum")
    btn_phase = QtWidgets.QPushButton("Phase-only")
    btn_magnitude = QtWidgets.QPushButton("Magnitude-only")
    btn_lowpass = QtWidgets.QPushButton("Low-pass")
    btn_highpass = QtWidgets.QPushButton("High-pass")
    btn_bandpass = QtWidgets.QPushButton("Band-pass")
    btn_notch = QtWidgets.QPushButton("Notch Remove")
    
    h_actions.addWidget(btn_spectrum)
    h_actions.addWidget(btn_phase)
    h_actions.addWidget(btn_magnitude)
    h_actions.addWidget(btn_lowpass)
    h_actions.addWidget(btn_highpass)
    h_actions.addWidget(btn_bandpass)
    h_actions.addWidget(btn_notch)
    
    layout.addWidget(grp_actions)
    
    # Results sub-tabs
    sub_tabs = QtWidgets.QTabWidget()
    
    # Create canvases for each result
    canvas_spectrum = ImageCanvas()
    canvas_phase = ImageCanvas()
    canvas_magnitude = ImageCanvas()
    canvas_lowpass = ImageCanvas()
    canvas_highpass = ImageCanvas()
    canvas_bandpass = ImageCanvas()
    canvas_notch = ImageCanvas()
    
    # Add to sub-tabs
    sub_tabs.addTab(canvas_spectrum, "Spectrum")
    sub_tabs.addTab(canvas_phase, "Phase-only")
    sub_tabs.addTab(canvas_magnitude, "Magnitude-only")
    sub_tabs.addTab(canvas_lowpass, "Low-pass")
    sub_tabs.addTab(canvas_highpass, "High-pass")
    sub_tabs.addTab(canvas_bandpass, "Band-pass")
    sub_tabs.addTab(canvas_notch, "Notch Remove")
    
    layout.addWidget(sub_tabs, 1)
    
    # Store references for easy access
    tab_fourier.cmb_kind = cmb_kind
    tab_fourier.spn_d0 = spn_d0
    tab_fourier.spn_n = spn_n
    tab_fourier.spn_d_low = spn_d_low
    tab_fourier.spn_d_high = spn_d_high
    tab_fourier.edt_notch = edt_notch
    tab_fourier.spn_radius = spn_radius
    tab_fourier.chk_rgb = chk_rgb
    tab_fourier.sub_tabs = sub_tabs
    tab_fourier.canvas_spectrum = canvas_spectrum
    tab_fourier.canvas_phase = canvas_phase
    tab_fourier.canvas_magnitude = canvas_magnitude
    tab_fourier.canvas_lowpass = canvas_lowpass
    tab_fourier.canvas_highpass = canvas_highpass
    tab_fourier.canvas_bandpass = canvas_bandpass
    tab_fourier.canvas_notch = canvas_notch
    
    # Connect buttons
    btn_spectrum.clicked.connect(lambda: on_spectrum_clicked(parent))
    btn_phase.clicked.connect(lambda: on_phase_clicked(parent))
    btn_magnitude.clicked.connect(lambda: on_magnitude_clicked(parent))
    btn_lowpass.clicked.connect(lambda: on_lowpass_clicked(parent))
    btn_highpass.clicked.connect(lambda: on_highpass_clicked(parent))
    btn_bandpass.clicked.connect(lambda: on_bandpass_clicked(parent))
    btn_notch.clicked.connect(lambda: on_notch_clicked(parent))
    
    return tab_fourier

def get_source_image(parent):
    """Get the source image for Fourier processing."""
    if hasattr(parent, '_after_rgb') and parent._after_rgb is not None:
        return parent._after_rgb
    elif hasattr(parent, '_before_rgb') and parent._before_rgb is not None:
        return parent._before_rgb
    else:
        return None

def on_spectrum_clicked(parent):
    """Handle spectrum button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        spectrum = compute_spectrum(img)
        parent.tabFourier.canvas_spectrum.set_image_rgb(spectrum)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_spectrum)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Spectrum computation failed: {str(e)}")

def on_phase_clicked(parent):
    """Handle phase-only button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        phase_result = phase_only_reconstruction(img)
        parent.tabFourier.canvas_phase.set_image_rgb(phase_result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_phase)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Phase-only reconstruction failed: {str(e)}")

def on_magnitude_clicked(parent):
    """Handle magnitude-only button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        magnitude_result = magnitude_only_reconstruction(img)
        parent.tabFourier.canvas_magnitude.set_image_rgb(magnitude_result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_magnitude)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Magnitude-only reconstruction failed: {str(e)}")

def on_lowpass_clicked(parent):
    """Handle low-pass filter button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        kind = parent.tabFourier.cmb_kind.currentText()
        D0 = parent.tabFourier.spn_d0.value()
        n = parent.tabFourier.spn_n.value()
        apply_rgb = parent.tabFourier.chk_rgb.isChecked()
        
        # Create filter
        if kind == "gaussian":
            H = gaussian_lowpass(img.shape[:2], D0)
        elif kind == "butterworth":
            H = butterworth_lowpass(img.shape[:2], D0, n)
        elif kind == "ideal":
            H = ideal_lowpass(img.shape[:2], D0)
        
        # Apply filter
        if apply_rgb and img.ndim == 3:
            result = apply_filter_rgb(img, H)
        else:
            result = apply_filter_gray(img, H)
        
        parent.tabFourier.canvas_lowpass.set_image_rgb(result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_lowpass)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Low-pass filter failed: {str(e)}")

def on_highpass_clicked(parent):
    """Handle high-pass filter button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        kind = parent.tabFourier.cmb_kind.currentText()
        D0 = parent.tabFourier.spn_d0.value()
        n = parent.tabFourier.spn_n.value()
        apply_rgb = parent.tabFourier.chk_rgb.isChecked()
        
        # Create filter
        if kind == "gaussian":
            H = gaussian_highpass(img.shape[:2], D0)
        elif kind == "butterworth":
            H = butterworth_highpass(img.shape[:2], D0, n)
        elif kind == "ideal":
            H = ideal_highpass(img.shape[:2], D0)
        
        # Apply filter
        if apply_rgb and img.ndim == 3:
            result = apply_filter_rgb(img, H)
        else:
            result = apply_filter_gray(img, H)
        
        parent.tabFourier.canvas_highpass.set_image_rgb(result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_highpass)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"High-pass filter failed: {str(e)}")

def on_bandpass_clicked(parent):
    """Handle band-pass filter button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        kind = parent.tabFourier.cmb_kind.currentText()
        D_low = parent.tabFourier.spn_d_low.value()
        D_high = parent.tabFourier.spn_d_high.value()
        n = parent.tabFourier.spn_n.value()
        apply_rgb = parent.tabFourier.chk_rgb.isChecked()
        
        # Create filter
        H = bandpass_filter(img.shape[:2], D_low, D_high, kind, n)
        
        # Apply filter
        if apply_rgb and img.ndim == 3:
            result = apply_filter_rgb(img, H)
        else:
            result = apply_filter_gray(img, H)
        
        parent.tabFourier.canvas_bandpass.set_image_rgb(result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_bandpass)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Band-pass filter failed: {str(e)}")

def on_notch_clicked(parent):
    """Handle notch filter button click."""
    img = get_source_image(parent)
    if img is None:
        QtWidgets.QMessageBox.information(parent, "Info", "Load/process an image first.")
        return
    
    try:
        notch_centers_str = parent.tabFourier.edt_notch.text().strip()
        radius = parent.tabFourier.spn_radius.value()
        apply_rgb = parent.tabFourier.chk_rgb.isChecked()
        
        # Parse notch centers
        centers = [c.strip() for c in notch_centers_str.split(';') if c.strip()]
        if not centers:
            QtWidgets.QMessageBox.warning(parent, "Warning", "No notch centers specified.")
            return
        
        # Create filter
        H = notch_reject_filter(img.shape[:2], centers, radius)
        
        # Apply filter
        if apply_rgb and img.ndim == 3:
            result = apply_filter_rgb(img, H)
        else:
            result = apply_filter_gray(img, H)
        
        parent.tabFourier.canvas_notch.set_image_rgb(result)
        parent.tabFourier.sub_tabs.setCurrentWidget(parent.tabFourier.canvas_notch)
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Notch filter failed: {str(e)}")