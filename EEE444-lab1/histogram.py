import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets


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
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No histogram to export.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Histogram", "histogram.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Exported to:\n{file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def close_tab(self):
        if self.main_window is None:
            self.close()
            return
        tabs = self.main_window.centerTabs
        idx = tabs.indexOf(self)
        if idx != -1:
            tabs.removeTab(idx)
        # clear stored reference if present
        if hasattr(self.main_window, '_histogram_tab') and self.main_window._histogram_tab is self:
            delattr(self.main_window, '_histogram_tab')

    def plot(self, image, title="Image Histogram", threshold=128):
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
    """Show histogram for the current image inside a new tab next to 'Processed'.

    If a histogram tab already exists it will be updated and selected.
    """
    try:
        image = main_window._after_rgb if main_window._after_rgb is not None else main_window._before_rgb
        if image is None:
            raise ValueError("No image loaded. Please load an image first.")

        image_type = "Grayscale" if calculate_histogram(image)['type'] == 'gray' else "RGB"
        title = f"{'Processed' if main_window._after_rgb is not None else 'Original'} Image Histogram ({image_type})"

        # If a histogram tab exists, update it
        if hasattr(main_window, '_histogram_tab') and main_window._histogram_tab is not None:
            tab = main_window._histogram_tab
            try:
                tab.plot(image, title=title, threshold=threshold)
            except Exception:
                # if update fails, recreate the tab
                tab = HistogramTab(image, title=title, threshold=threshold, main_window=main_window)
                # replace existing tab
                tabs = main_window.centerTabs
                idx = tabs.indexOf(main_window._histogram_tab)
                if idx != -1:
                    tabs.removeTab(idx)
                    tabs.insertTab(idx, tab, "Histogram")
                main_window._histogram_tab = tab
        else:
            tab = HistogramTab(image, title=title, threshold=threshold, main_window=main_window)
            # insert tab after Processed (which is index 1 if Original=0, Processed=1)
            tabs = main_window.centerTabs
            insert_idx = 2 if tabs.count() >= 2 else tabs.count()
            tabs.insertTab(insert_idx, tab, "Histogram")
            main_window._histogram_tab = tab

        # make the histogram tab active
        tabs.setCurrentWidget(main_window._histogram_tab)

    except ValueError as e:
        QtWidgets.QMessageBox.information(main_window, "Info", str(e))
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
