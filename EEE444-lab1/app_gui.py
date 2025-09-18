import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from my_image_processing import MyProcess2
from binary import convert_to_binary
from zoom import ZoomHandler

APP_TITLE = "DIP Lab — Image Studio"

# -------------------------- Dark QSS Theme --------------------------
DARK_QSS = """
* { font-family: 'Segoe UI', Arial; font-size: 10.5pt; color: #E6E6E6; }
QWidget { background-color: #1f2023; }
QFrame#SidePanel { background-color: #2b2d31; border: 1px solid #2b2d31; border-radius: 8px; }
QFrame#TopBar { background-color: #2b2d31; border: 1px solid #2b2d31; border-radius: 8px; }
QTabWidget::pane { border: 1px solid #2b2d31; top: -0.5em; background: #1f2023; }
QTabBar::tab { background: #2b2d31; border: 1px solid #2b2d31; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #3a3c42; }
QTabBar::tab:hover { background: #34363b; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit { background: #1c1d21; border: 1px solid #3a3c42; padding: 6px; border-radius: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QPushButton { background-color: #f1c57a; color: #2b2d31; border: 0px; padding: 8px 14px; border-radius: 8px; font-weight: 600; }
QPushButton:hover { background-color: #ffd28e; }
QPushButton:pressed { background-color: #e7b86a; }
QPushButton#Secondary { background: #3a3c42; color: #E6E6E6; font-weight: 500; }
QPushButton#Secondary:hover { background: #44474e; }
QListWidget, QTreeWidget { background: #1c1d21; border: 1px solid #3a3c42; border-radius: 6px; }
QSlider::groove:horizontal { height: 6px; background: #3a3c42; border-radius: 3px; }
QSlider::handle:horizontal { background: #f1c57a; width: 14px; border-radius: 7px; margin: -4px 0; }
"""

# -------------------------- Helpers --------------------------
def np_rgb_to_qpixmap(img_rgb: np.ndarray, target_size: QtCore.QSize) -> QtGui.QPixmap:
    """Convert RGB ndarray -> QPixmap, scaled with aspect ratio."""
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

# -------------------------- Main Window --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1300, 750)
        self._before_rgb = None
        self._after_rgb = None

        # ---- Root containers: Top bar + Splitter (side panel | central tabs | right tabs) ----
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # Top bar
        self.topBar = QtWidgets.QFrame(objectName="TopBar")
        top_lay = QtWidgets.QHBoxLayout(self.topBar)
        top_lay.setContentsMargins(12, 8, 12, 8)
        self.titleLabel = QtWidgets.QLabel("Image Analysis")
        self.titleLabel.setStyleSheet("font-size: 13pt; font-weight:700;")
        top_lay.addWidget(self.titleLabel)
        top_lay.addStretch(1)
        self.btnOpen = QtWidgets.QPushButton("Load Image")
        self.btnSave = QtWidgets.QPushButton("Export Result")
        self.btnOpen.setObjectName("Secondary")
        self.btnSave.setObjectName("Secondary")
        top_lay.addWidget(self.btnOpen)
        top_lay.addWidget(self.btnSave)
        root_layout.addWidget(self.topBar)

        # Splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter, 1)

        # ---- Left side panel ----
        self.sidePanel = QtWidgets.QFrame(objectName="SidePanel")
        side = QtWidgets.QVBoxLayout(self.sidePanel)
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)

        # logo placeholder
        logo = QtWidgets.QLabel("DIP LAB")
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo.setStyleSheet("font-size: 16pt; font-weight: 800; color: #f1c57a;")
        side.addWidget(logo)

        self.stepList = QtWidgets.QListWidget()
        self.stepList.addItems([
            "1. Επιλογή Εικόνας",
            "2. Διαστάσεις (resize)",
            "3. Περιστροφή (deg)",
            "4. Negative (προαιρετικό)",
            "5. Εκτέλεση",
            "6. Αποθήκευση",
            "7. Ιστόγραμμα (προαιρετικό)",
            "8. Binary (προαιρετικό)"
        ])
        self.stepList.setFixedWidth(260)
        side.addWidget(self.stepList, 1)

        # Controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.imagesDir = QtWidgets.QLineEdit("./images")
        self.subjectSpin = QtWidgets.QSpinBox()
        self.subjectSpin.setRange(1, 999)
        self.subjectSpin.setValue(1)
        self.imagePath = QtWidgets.QLineEdit()
        self.imagePath.setPlaceholderText("ή άφησέ το κενό για subject<x>.jpg")
        self.btnPick = QtWidgets.QPushButton("Επιλογή Εικόνας")
        self.btnPick.setObjectName("Secondary")
        self.resizeW = QtWidgets.QLineEdit()
        self.resizeW.setPlaceholderText("πλάτος (ή κενό)")
        self.resizeH = QtWidgets.QLineEdit()
        self.resizeH.setPlaceholderText("ύψος (ή κενό)")
        self.smallDim = QtWidgets.QLineEdit()
        self.smallDim.setPlaceholderText("μικρή διάσταση (int)")
        self.angle = QtWidgets.QLineEdit("0")
        self.chkNegative = QtWidgets.QCheckBox("Negative")
        self.chkBinary = QtWidgets.QCheckBox("Binary") 

        # Binary slider + label
        self.binaryThresholdSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.binaryThresholdSlider.setRange(0, 255)
        self.binaryThresholdSlider.setValue(127)
        self.binaryThresholdSlider.setFixedWidth(120)
        self.binaryThresholdLabel = QtWidgets.QLabel("127")
        self.binaryThresholdLabel.setFixedWidth(30)
        self.binaryThresholdSlider.valueChanged.connect(
            lambda v: self.binaryThresholdLabel.setText(str(v))
        )

        binaryRow = QtWidgets.QHBoxLayout()
        binaryRow.addWidget(self.chkBinary)
        binaryRow.addWidget(self.binaryThresholdSlider)
        binaryRow.addWidget(self.binaryThresholdLabel)

        form.addRow("images_dir:", self.imagesDir)
        form.addRow("subject_index:", self.subjectSpin)
        form.addRow("image path:", self.imagePath)
        form.addRow("", self.btnPick)
        form.addRow("resize width:", self.resizeW)
        form.addRow("resize height:", self.resizeH)
        form.addRow("resize small-dim:", self.smallDim)
        form.addRow("angle (deg):", self.angle)
        form.addRow("", self.chkNegative)
        form.addRow("", binaryRow)
        side.addLayout(form)

        # action buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.btnLoadSubject = QtWidgets.QPushButton("Load subject<x>.jpg")
        self.btnRun = QtWidgets.QPushButton("Apply")
        btnRow.addWidget(self.btnLoadSubject)
        btnRow.addWidget(self.btnRun)
        side.addLayout(btnRow)
        splitter.addWidget(self.sidePanel)

        # ---- Center: Tabs with images ----
        self.centerTabs = QtWidgets.QTabWidget()
        self.tabOriginal = QtWidgets.QWidget()
        self.tabProcessed = QtWidgets.QWidget()
        self.centerTabs.addTab(self.tabOriginal, "Original")
        self.centerTabs.addTab(self.tabProcessed, "Processed")

        # Original area
        oLay = QtWidgets.QVBoxLayout(self.tabOriginal)
        oLay.setContentsMargins(8, 8, 8, 8)
        self.lblBefore = QtWidgets.QLabel("— καμία εικόνα —")
        self.lblBefore.setAlignment(QtCore.Qt.AlignCenter)
        self.lblBefore.setMinimumSize(560, 420)
        self.lblBefore.setStyleSheet("border: 1px dashed #3a3c42;")
        oLay.addWidget(self.lblBefore, 1)

        # Processed area
        pLay = QtWidgets.QVBoxLayout(self.tabProcessed)
        pLay.setContentsMargins(8, 8, 8, 8)
        self.lblAfter = QtWidgets.QLabel("— εκτέλεσε για να δεις αποτέλεσμα —")
        self.lblAfter.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAfter.setMinimumSize(560, 420)
        self.lblAfter.setStyleSheet("border: 1px dashed #3a3c42;")
        pLay.addWidget(self.lblAfter, 1)
        splitter.addWidget(self.centerTabs)

        # ---- Right: Tabs for "Operations" ----
        self.rightTabs = QtWidgets.QTabWidget()
        self.rightTabs.addTab(self._build_operations_tab(), "Operations")
        self.rightTabs.addTab(self._build_info_tab(), "Info")
        splitter.addWidget(self.rightTabs)

        # initial sizes
        splitter.setSizes([320, 700, 300])

        # ---- Zoom controls ----
        zoomRow = QtWidgets.QHBoxLayout()
        zoomLabel = QtWidgets.QLabel("Zoom:")
        self.zoomSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoomSlider.setRange(10, 300)  
        self.zoomSlider.setValue(100)
        self.zoomSlider.setFixedWidth(120)   
        self.zoomValueLabel = QtWidgets.QLabel("100%")
        self.zoomValueLabel.setFixedWidth(40) 

        zoomRow.addWidget(zoomLabel)
        zoomRow.addWidget(self.zoomSlider)
        zoomRow.addWidget(self.zoomValueLabel)
        zoomRow.addStretch(1)  
        root_layout.addLayout(zoomRow)

        # Zoom Handlers
        self.zoomHandlerBefore = ZoomHandler(self.lblBefore)
        self.zoomHandlerAfter = ZoomHandler(self.lblAfter)

        # ---- Signals ----
        self.btnPick.clicked.connect(self.on_pick_image)
        self.btnOpen.clicked.connect(self.on_pick_image)
        self.btnLoadSubject.clicked.connect(self.on_load_subject)
        self.btnRun.clicked.connect(self.on_apply)
        self.btnSave.clicked.connect(self.on_save)
        self.zoomSlider.valueChanged.connect(self.on_zoom_changed)

        # apply style
        self.setStyleSheet(DARK_QSS)

    # Right tabs content
    def _build_operations_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        info = QtWidgets.QLabel(
            "• Ορισμοί:\n"
            " – 'resize small-dim': αν συμπληρωθεί (ακέραιος), αγνοούνται width/height.\n"
            " – width/height: μπορείτε να αφήσετε ένα κενό για διατήρηση αναλογιών.\n"
            " – angle: μοίρες περιστροφής (αριστερόστροφα).\n"
            " – Negative: αντιστρέφει τα χρώματα.\n"
            " – Binary: μετατρέπει την εικόνα σε δυαδική (threshold=slider)."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addStretch(1)
        return w

    def _build_info_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lbl = QtWidgets.QLabel(
            "DIP Lab — Image Studio\n"
            "PyQt5/OpenCV demo UI.\n"
            "Φόρτωσε εικόνα, όρισε μετασχηματισμούς και αποθήκευσε αποτέλεσμα.\n"
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addStretch(1)
        return w

    # -------------------------- Actions --------------------------
    def on_pick_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Επιλογή Εικόνας", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        self.imagePath.setText(path)
        self._load_to_left(path)

    def on_load_subject(self):
        images_dir = self.imagesDir.text().strip() or "./images"
        idx = self.subjectSpin.value()
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        found_path = None
        for ext in exts:
            p = os.path.join(images_dir, f"subject{idx}{ext}")
            if os.path.exists(p):
                found_path = p
                break
        if not found_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Προσοχή",
                f"Δεν βρέθηκε αρχείο subject{idx} με επεκτάσεις {exts} στον φάκελο:\n{images_dir}"
            )
            return
        self.imagePath.clear()
        self._load_to_left(found_path)

    def _load_to_left(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", "Αποτυχία φόρτωσης εικόνας.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._before_rgb = rgb
        pix = np_rgb_to_qpixmap(rgb, self.lblBefore.size())
        self.zoomHandlerBefore.set_pixmap(pix)
        self.centerTabs.setCurrentWidget(self.tabOriginal)

    def _parse_resize(self):
        sd = self.smallDim.text().strip()
        if sd:
            return int(sd)
        wtxt = self.resizeW.text().strip()
        htxt = self.resizeH.text().strip()
        w = int(wtxt) if wtxt else None
        h = int(htxt) if htxt else None
        if w is None and h is None:
            return None
        return (w, h)

    def on_apply(self):
        try:
            resize_arg = self._parse_resize()
            angle = float(self.angle.text().strip()) if self.angle.text().strip() else None
            negative = self.chkNegative.isChecked()
            images_dir = self.imagesDir.text().strip() or "./images"
            subject_index = self.subjectSpin.value() if not self.imagePath.text().strip() else None
            image_path = self.imagePath.text().strip() or None
            out_rgb = MyProcess2(
                image_path=image_path,
                subject_index=subject_index,
                images_dir=images_dir,
                resize=resize_arg,
                rotate=angle,
                negative=negative,
                show=False,
                out_path=None
            )
            # Apply binary conversion if selected
            if self.chkBinary.isChecked():
                threshold = self.binaryThresholdSlider.value()
                out_rgb = convert_to_binary(out_rgb, threshold=threshold)
            self._after_rgb = out_rgb
            pix = np_rgb_to_qpixmap(out_rgb, self.lblAfter.size())
            self.zoomHandlerAfter.set_pixmap(pix)
            self.centerTabs.setCurrentWidget(self.tabProcessed)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", str(e))

    def on_save(self):
        if self._after_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Δεν υπάρχει αποτέλεσμα για αποθήκευση.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Αποθήκευση", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        bgr = cv2.cvtColor(self._after_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        QtWidgets.QMessageBox.information(self, "OK", f"Αποθηκεύτηκε: {path}")

    def on_zoom_changed(self, value: int):
        self.zoomValueLabel.setText(f"{value}%")
        self.zoomHandlerBefore.set_zoom(value)
        self.zoomHandlerAfter.set_zoom(value)

# -------------------------- Run App --------------------------
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
