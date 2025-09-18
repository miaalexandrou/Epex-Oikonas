from PyQt5 import QtCore, QtGui, QtWidgets

class ZoomHandler:
    def __init__(self, label: QtWidgets.QLabel):
        self.label = label
        self.original_pixmap = None
        self.current_scale = 100  # percentage

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        """Store the original pixmap and display it."""
        self.original_pixmap = pixmap
        self.apply_zoom()

    def set_zoom(self, value: int):
        """Update zoom level (percentage)."""
        self.current_scale = value
        self.apply_zoom()

    def apply_zoom(self):
        """Rescale and update QLabel pixmap."""
        if not self.original_pixmap:
            return
        w = self.original_pixmap.width() * self.current_scale // 100
        h = self.original_pixmap.height() * self.current_scale // 100
        scaled = self.original_pixmap.scaled(
            w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)
