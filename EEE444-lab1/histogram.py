import io
import datetime
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets


def _resize_and_pad_rgb(img_rgb, size=(600, 400), pad_color=(255, 255, 255)):
    """Resize RGB image to fit into size while preserving aspect ratio and pad with pad_color.

    Args:
        img_rgb: HxWx3 uint8 RGB image (numpy).
        size: (w, h) target box.
        pad_color: RGB tuple.
    Returns:
        image of shape (h, w, 3) uint8 RGB.
    """
    if img_rgb is None:
        return np.full((size[1], size[0], 3), pad_color, dtype=np.uint8)

    h, w = img_rgb.shape[:2]
    tgt_w, tgt_h = size
    scale = min(tgt_w / w, tgt_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # create background
    bg = np.full((tgt_h, tgt_w, 3), pad_color, dtype=np.uint8)
    # center
    x = (tgt_w - new_w) // 2
    y = (tgt_h - new_h) // 2
    bg[y:y+new_h, x:x+new_w] = resized
    return bg


def _render_figure_to_rgb_array(figure):
    """Render a Matplotlib Figure to an RGB uint8 numpy array."""
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Convert BGR(A) -> RGB
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _render_steps_panel(main_window, size=(600, 400), pad_color=(255, 255, 255)):
    """Create an RGB panel with the processing steps read from main_window controls."""
    w, h = size
    panel = np.full((h, w, 3), pad_color, dtype=np.uint8)
    lines = []
    try:
        # source info
        src = None
        if getattr(main_window, 'imagePath', None) and main_window.imagePath.text().strip():
            src = main_window.imagePath.text().strip()
        else:
            images_dir = main_window.imagesDir.text().strip() or './images'
            idx = main_window.subjectSpin.value() if getattr(main_window, 'subjectSpin', None) else None
            src = f"{images_dir}/subject{idx}.jpg" if idx is not None else images_dir
        lines.append(f"Source: {src}")

        # resize
        sd = main_window.smallDim.text().strip() if getattr(main_window, 'smallDim', None) else ''
        wtxt = main_window.resizeW.text().strip() if getattr(main_window, 'resizeW', None) else ''
        htxt = main_window.resizeH.text().strip() if getattr(main_window, 'resizeH', None) else ''
        if sd:
            lines.append(f"Resize (small-dim): {sd}")
        else:
            lines.append(f"Resize width: {wtxt or 'auto'}, height: {htxt or 'auto'}")

        # rotate
        angle = main_window.angle.text().strip() if getattr(main_window, 'angle', None) else '0'
        lines.append(f"Rotate: {angle} deg")

        # negative
        neg = main_window.chkNegative.isChecked() if getattr(main_window, 'chkNegative', None) else False
        lines.append(f"Negative: {neg}")

        # binary
        binary = main_window.chkBinary.isChecked() if getattr(main_window, 'chkBinary', None) else False
        bthr = main_window.binaryThresholdSlider.value() if getattr(main_window, 'binaryThresholdSlider', None) else None
        lines.append(f"Binary: {binary}" + (f" (thr={bthr})" if bthr is not None else ""))

        # timestamp
        lines.append(f"Exported: {datetime.datetime.now().isoformat(timespec='seconds')}")
    except Exception:
        lines = ["(no processing metadata available)"]

    # render lines with OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (20, 20, 20)  # dark text on white
    line_height = int(22 * font_scale) + 18
    x = 16
    y = 32
    for line in lines:
        # wrap long lines
        max_chars = 60
        parts = [line[i:i+max_chars] for i in range(0, len(line), max_chars)]
        for p in parts:
            cv2.putText(panel, p, (x, y), font, font_scale, color, thickness=1, lineType=cv2.LINE_AA)
            y += line_height
            if y > h - 20:
                break
        if y > h - 20:
            break

    return panel


def calculate_histogram(image):
    """Calculate histogram for RGB or grayscale image."""
    is_gray = len(image.shape) == 2 or (len(image.shape) == 3 and
                np.array_equal(image[:, :, 0], image[:, :, 1]) and
                np.array_equal(image[:, :, 1], image[:, :, 2]))

    if is_gray:
        gray = image[:, :, 0] if len(image.shape) == 3 else image
        return {'type': 'gray', 'gray': cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()}
    else:
        # assume image is RGB (0=R,1=G,2=B)
        return {
            'type': 'rgb',
            'red': cv2.calcHist([image], [0], None, [256], [0, 256]).flatten(),
            'green': cv2.calcHist([image], [1], None, [256], [0, 256]).flatten(),
            'blue': cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        }


def calculate_stats(hist_data):
    """Calculate basic statistics from histogram."""
    stats = {}
    x = np.arange(256)

    channels = ['gray'] if hist_data['type'] == 'gray' else ['red', 'green', 'blue']
    for ch in channels:
        data = hist_data[ch]
        total = np.sum(data)
        if total > 0:
            mean = np.sum(x * data) / total
            std = np.sqrt(np.sum(((x - mean) ** 2) * data) / total)
            stats[ch] = {'mean': mean, 'std': std}
    return stats


class HistogramTab(QtWidgets.QWidget):
    """A QWidget containing a matplotlib FigureCanvas to be added as a tab.

    This widget supports updating the histogram, exporting the figure, and closing
    itself (which removes the tab from the main window's centerTabs).
    """

    def __init__(self, image, title="Histogram", threshold=128, main_window=None):
        super().__init__()
        self.setObjectName('HistogramTab')
        self.main_window = main_window
        self.current_image = None
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

        btn_layout = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export")
        export_btn.clicked.connect(self.export_histogram)
        close_btn = QtWidgets.QPushButton("Close Tab")
        close_btn.clicked.connect(self.close_tab)
        btn_layout.addStretch(1)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # initial plot
        self.plot(image, title=title, threshold=threshold)

    def export_histogram(self):
        """Export a composed image containing:
        top-left: original image
        top-right: processed image
        bottom-left: histogram (this figure)
        bottom-right: text panel with processing steps

        The composed image is saved to the selected file path.
        """
        main_win = self.main_window
        if main_win is None:
            # fallback to simple figure export
            super_export = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export Histogram", "histogram.png",
                "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            file_path = super_export[0]
            if not file_path:
                return
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Exported to:\n{file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
            return

        if main_win._before_rgb is None and main_win._after_rgb is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No images available to export.")
            return

        # choose save path
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Summary Image", "summary.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if not file_path:
            return

        try:
            # prepare panels
            orig = None if getattr(main_win, '_before_rgb', None) is None else main_win._before_rgb
            proc = None if getattr(main_win, '_after_rgb', None) is None else main_win._after_rgb

            # render histogram figure to RGB array
            hist_rgb = _render_figure_to_rgb_array(self.figure)

            # resize/pad each to same target panel size
            panel_size = (640, 480)
            p_orig = _resize_and_pad_rgb(orig, size=panel_size, pad_color=(255, 255, 255))
            p_proc = _resize_and_pad_rgb(proc, size=panel_size, pad_color=(255, 255, 255))
            p_hist = _resize_and_pad_rgb(hist_rgb, size=panel_size, pad_color=(255, 255, 255))
            p_steps = _render_steps_panel(main_win, size=panel_size, pad_color=(255, 255, 255))

            # compose 2x2
            top = np.hstack([p_orig, p_proc])
            bottom = np.hstack([p_hist, p_steps])
            canvas = np.vstack([top, bottom])

            # write out using OpenCV (BGR)
            bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(file_path, bgr)
            if not ok:
                raise IOError("OpenCV failed to write file")

            QtWidgets.QMessageBox.information(self, "Success", f"Exported summary to:\n{file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def close_tab(self):
        if self.main_window is None:
            self.close()
            return
            
        # Check if this is the permanent histogram tab
        if hasattr(self.main_window, 'tabHistogram') and self.main_window.tabHistogram is self:
            QtWidgets.QMessageBox.information(self, "Info", "This is a permanent tab and cannot be closed.")
            return
            
        # Only allow closing if it's not the permanent tab
        tabs = self.main_window.centerTabs
        idx = tabs.indexOf(self)
        if idx != -1:
            tabs.removeTab(idx)
        # clear stored reference if present
        if hasattr(self.main_window, '_histogram_tab') and self.main_window._histogram_tab is self:
            delattr(self.main_window, '_histogram_tab')

    def plot(self, image, title="Image Histogram", threshold=128):
        if image is None:
            # Handle case when no image is provided
            self.current_image = None
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No Image Loaded\nLoad an image to see histogram', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_xlim([0, 255])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            self.figure.tight_layout()
            self.canvas.draw()
            return
            
        hist_data = calculate_histogram(image)
        stats = calculate_stats(hist_data)

        self.current_image = image.copy()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = np.arange(256)

        if hist_data['type'] == 'rgb':
            colors = ['red', 'green', 'blue']
            for color in colors:
                ax.plot(x, hist_data[color], color=color, alpha=0.8, linewidth=2, label=color.capitalize())
            stats_text = "\n".join([f"{c[0].upper()}: μ={stats[c]['mean']:.1f}, σ={stats[c]['std']:.1f}"
                                     for c in colors if c in stats])
            ax.set_title(f'{title} - RGB')
            ax.legend()
        else:
            ax.fill_between(x, hist_data['gray'], alpha=0.6, color='lightgray')
            ax.plot(x, hist_data['gray'], color='black', linewidth=2)
            stats_text = f"Mean: {stats['gray']['mean']:.1f}\nStd: {stats['gray']['std']:.1f}" if 'gray' in stats else ""
            ax.set_title(f'{title} - Grayscale')

        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 255])
        ax.set_facecolor('#f8f9fa')

        # Add threshold line
        ax.axvline(x=threshold, color='red', linewidth=2, linestyle='-', label=f'Threshold: {threshold}')

        if stats_text:
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.figure.tight_layout()
        self.canvas.draw()


def show_histogram_gui(main_window, threshold=128):
    """Update and show the permanent histogram tab.
    
    This function now works with the permanent histogram tab instead of creating new ones.
    """
    try:
        image = main_window._after_rgb if main_window._after_rgb is not None else main_window._before_rgb
        if image is None:
            QtWidgets.QMessageBox.information(main_window, "Info", "No image loaded. Please load an image first.")
            return

        image_type = "Grayscale" if calculate_histogram(image)['type'] == 'gray' else "RGB"
        title = f"{'Processed' if main_window._after_rgb is not None else 'Original'} Image Histogram ({image_type})"

        # Update the permanent histogram tab
        if hasattr(main_window, 'tabHistogram') and main_window.tabHistogram is not None:
            main_window.tabHistogram.plot(image, title=title, threshold=threshold)
            # Switch to histogram tab
            main_window.centerTabs.setCurrentWidget(main_window.tabHistogram)
        else:
            QtWidgets.QMessageBox.warning(main_window, "Warning", "Histogram tab not found.")

    except Exception as e:
        QtWidgets.QMessageBox.critical(main_window, "Error", f"Failed to generate histogram: {str(e)}")


def compare_histograms(image1, image2, title1="Image 1", title2="Image 2"):
    """Create a tab widget that compares two image histograms side-by-side.

    Returns: HistogramTab (with a modified figure showing comparison)
    """
    tab = HistogramTab(image1, title=title1, threshold=128, main_window=None)
    tab.figure.clear()
    ax = tab.figure.add_subplot(111)
    hist1, hist2 = calculate_histogram(image1), calculate_histogram(image2)
    x = np.arange(256)

    if hist1['type'] == 'rgb':
        for color in ['red', 'green', 'blue']:
            ax.plot(x, hist1[color], color=color, alpha=0.8, linewidth=2, label=f'{title1} - {color.capitalize()}')
    else:
        ax.plot(x, hist1['gray'], color='black', alpha=0.8, linewidth=2, label=f'{title1} - Gray')

    if hist2['type'] == 'rgb':
        for color in ['red', 'green', 'blue']:
            ax.plot(x, hist2[color], color=color, alpha=0.6, linewidth=2, linestyle='--', label=f'{title2} - {color.capitalize()}')
    else:
        ax.plot(x, hist2['gray'], color='gray', alpha=0.6, linewidth=2, linestyle='--', label=f'{title2} - Gray')

    ax.set_title('Histogram Comparison')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 255])
    ax.set_facecolor('#f8f9fa')

    tab.figure.tight_layout()
    tab.canvas.draw()
    return tab
