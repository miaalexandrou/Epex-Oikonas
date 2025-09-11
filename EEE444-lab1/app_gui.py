import os, cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from my_image_processing import MyProcess2, DEFAULT_IMAGES_DIR
from binary import convert_to_binary

APP_TITLE = "DIP Lab — Image Studio"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1000, 600)
        self._before = None
        self._after = None
        
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        
        # Buttons
        self.btnOpen = QtWidgets.QPushButton("Load Image")
        self.btnRun = QtWidgets.QPushButton("Apply")
        self.btnBinary = QtWidgets.QPushButton("Make Binary")  # New button
        self.btnSave = QtWidgets.QPushButton("Save Result")
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.btnOpen)
        hl.addWidget(self.btnRun)
        hl.addWidget(self.btnBinary)  # Add binary button
        hl.addWidget(self.btnSave)
        layout.addLayout(hl)
        
        # Images
        self.lblBefore = QtWidgets.QLabel("Before")
        self.lblAfter = QtWidgets.QLabel("After")
        self.lblBefore.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAfter.setAlignment(QtCore.Qt.AlignCenter)
        hl2 = QtWidgets.QHBoxLayout()
        hl2.addWidget(self.lblBefore)
        hl2.addWidget(self.lblAfter)
        layout.addLayout(hl2)
        
        # Connect signals
        self.btnOpen.clicked.connect(self.on_open)
        self.btnRun.clicked.connect(self.on_apply)
        self.btnBinary.clicked.connect(self.on_binary)  # Connect binary button
        self.btnSave.clicked.connect(self.on_save)
    
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.jpg *.png)")
        if not path:
            return
        
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._before = rgb
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.lblBefore.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(400, 400, QtCore.Qt.KeepAspectRatio))
    
    def on_apply(self):
        if self._before is None:
            return
        
        self._after = MyProcess2(
            image_path=None,
            subject_index=1,
            images_dir=DEFAULT_IMAGES_DIR,
            resize=(320, 240),
            rotate=35,
            negative=True,
            show=False
        )
        
        h, w, ch = self._after.shape
        qimg = QtGui.QImage(self._after.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.lblAfter.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(400, 400, QtCore.Qt.KeepAspectRatio))
    
    def on_binary(self):
        """Handle binary conversion button click."""
        if self._before is None:
            return
            
        self._after = convert_to_binary(self._before)
        h, w, ch = self._after.shape
        qimg = QtGui.QImage(self._after.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.lblAfter.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(400, 400, QtCore.Qt.KeepAspectRatio))
    
    def on_save(self):
        if self._after is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "", "PNG (*.png)")
        if path:
            cv2.imwrite(path, cv2.cvtColor(self._after, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()