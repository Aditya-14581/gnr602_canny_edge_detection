import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QGridLayout, QPushButton, QLineEdit, QScrollArea, QGroupBox, QFormLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from scipy.ndimage import convolve
from PIL import Image

# Gaussian Kernel for blurring
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Apply Gaussian filter
def apply_gaussian_blur(img, sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(img, kernel)

# Sobel operator for edge detection
def sobel_filters(img):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve(img, Gx)
    grad_y = convolve(img, Gy)
    
    grad_magnitude = np.hypot(grad_x, grad_y)
    grad_angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) % 180  # Convert to degrees
    
    return grad_magnitude, grad_angle

# Non-maximum suppression
def non_maximum_suppression(magnitude, angle):
    rows, cols = magnitude.shape
    output = np.zeros_like(magnitude)
    angle = angle / 180.0 * np.pi  # Convert to radians
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Get the direction of the gradient
            direction = angle[i, j]
            
            # Determine the neighbors
            if (0 <= direction < np.pi / 8) or (15 * np.pi / 8 <= direction <= 2 * np.pi):
                neighbor1, neighbor2 = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (np.pi / 8 <= direction < 3 * np.pi / 8):
                neighbor1, neighbor2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (3 * np.pi / 8 <= direction < 5 * np.pi / 8):
                neighbor1, neighbor2 = magnitude[i + 1, j], magnitude[i - 1, j]
            else:
                neighbor1, neighbor2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            
            # Suppress non-maximal edges
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                output[i, j] = magnitude[i, j]
    
    return output

# Double thresholding and edge tracking
def double_threshold(magnitude, low_thresh, high_thresh):
    strong_edges = (magnitude > high_thresh)
    weak_edges = (magnitude >= low_thresh) & (magnitude <= high_thresh)
    
    return strong_edges, weak_edges

# Edge tracking by hysteresis
def edge_tracking(strong_edges, weak_edges):
    rows, cols = strong_edges.shape
    output = np.copy(strong_edges)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                if ((strong_edges[i + 1, j] or strong_edges[i - 1, j]) or 
                    (strong_edges[i, j + 1] or strong_edges[i, j - 1]) or 
                    (strong_edges[i + 1, j + 1] or strong_edges[i - 1, j - 1])):
                    output[i, j] = 1
    
    return output

# Main Canny edge detector
def canny_edge_detector(image_path, sigma, low_thresh, high_thresh):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Apply Gaussian blur
    blurred_img = apply_gaussian_blur(img, sigma)
    
    # Step 2: Calculate gradients
    magnitude, angle = sobel_filters(blurred_img)
    
    # Step 3: Non-maximum suppression
    suppressed_img = non_maximum_suppression(magnitude, angle)
    
    # Step 4: Double thresholding
    strong_edges, weak_edges = double_threshold(suppressed_img, low_thresh, high_thresh)
    
    # Step 5: Edge tracking by hysteresis
    final_edges = edge_tracking(strong_edges, weak_edges)
    
    return final_edges

from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt


class CannyEdgeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Canny Edge Detection')
        self.setGeometry(100, 100, 800, 600)
        
        self.selected_image_path = ''
        
        self.init_ui()

    def init_ui(self):
        # Set up layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Input for thresholds using sliders
        form_layout = QFormLayout()
        
        # Lower Threshold Slider
        self.low_threshold_slider = QSlider(Qt.Horizontal, self)
        self.low_threshold_slider.setRange(0, 100)  # Range from 0% to 100%
        self.low_threshold_slider.setValue(30)  # Default value (30%)
        self.low_threshold_slider.valueChanged.connect(self.update_low_threshold_label)

        # Lower Threshold Label
        self.low_threshold_label = QLabel(f'Low Threshold: {self.low_threshold_slider.value()}%', self)

        # Higher Threshold Slider
        self.high_threshold_slider = QSlider(Qt.Horizontal, self)
        self.high_threshold_slider.setRange(0, 100)  # Range from 0% to 100%
        self.high_threshold_slider.setValue(60)  # Default value (60%)
        self.high_threshold_slider.valueChanged.connect(self.update_high_threshold_label)

        # Higher Threshold Label
        self.high_threshold_label = QLabel(f'High Threshold: {self.high_threshold_slider.value()}%', self)

        # Add sliders and labels to the form layout
        form_layout.addRow(self.low_threshold_label, self.low_threshold_slider)
        form_layout.addRow(self.high_threshold_label, self.high_threshold_slider)
        layout.addLayout(form_layout)

        # Open image button
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)  # Connect to the open_image method
        layout.addWidget(self.open_button)

        # Apply edge detection button
        self.apply_button = QPushButton('Apply Edge Detection')
        self.apply_button.clicked.connect(self.apply_edge_detection)
        layout.addWidget(self.apply_button)

        # Scrollable area for images
        self.scroll_area = QScrollArea(self)
        layout.addWidget(self.scroll_area)

        # Scrollable widget for displaying images
        self.images_widget = QWidget()
        self.scroll_area.setWidget(self.images_widget)

        # Grid layout for images
        self.images_layout = QGridLayout(self.images_widget)
        
        self.scroll_area.setWidgetResizable(True)

    def open_image(self):
        # Open image file dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg);;All Files (*)', options=options)
        
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.images_layout.addWidget(label)

    def update_low_threshold_label(self):
        # Update label with current slider value for the lower threshold
        self.low_threshold_label.setText(f'Low Threshold: {self.low_threshold_slider.value()}%')

    def update_high_threshold_label(self):
        # Update label with current slider value for the higher threshold
        self.high_threshold_label.setText(f'High Threshold: {self.high_threshold_slider.value()}%')

    def apply_edge_detection(self):
        # Get percentage values from the sliders for thresholds
        low_thresh_percent = self.low_threshold_slider.value()
        high_thresh_percent = self.high_threshold_slider.value()

        # Convert percentages to actual pixel intensity values (0-255)
        low_thresh = int(low_thresh_percent * 255 / 100)
        high_thresh = int(high_thresh_percent * 255 / 100)

        if not self.selected_image_path:
            return
        
        # Clear previous images from layout
        for i in range(self.images_layout.count()):
            widget = self.images_layout.itemAt(i).widget()
            widget.deleteLater()

        # Loop over sigma values from 1 to 8
        for idx, sigma in enumerate(range(1, 9)):
            edges = canny_edge_detector(self.selected_image_path, sigma, low_thresh, high_thresh)

            # Convert edges to an image
            edges_image = Image.fromarray((edges * 255).astype(np.uint8))
            edges_image = edges_image.convert("RGB")
            
            # Resize to 400x400
            edges_image = edges_image.resize((400, 400))
            
            # Convert to QImage using QImage directly (no need for ImageQt)
            width, height = edges_image.size
            bytes_per_line = 3 * width
            img_data = edges_image.tobytes("raw", "RGB")
            edges_image_qt = QImage(img_data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert to QPixmap
            pixmap = QPixmap.fromImage(edges_image_qt)

            # Create a label for the image
            image_label = QLabel(self)
            image_label.setPixmap(pixmap)

            # Create a label for the sigma value
            sigma_label = QLabel(f'Sigma: {sigma}', self)

            # Create a label for the threshold percentages
            threshold_label = QLabel(f'Low Threshold: {low_thresh_percent}% | High Threshold: {high_thresh_percent}%', self)

            # Create a layout to stack the image, sigma value, and threshold labels vertically
            image_layout = QVBoxLayout()
            image_layout.addWidget(image_label)
            image_layout.addWidget(sigma_label)
            image_layout.addWidget(threshold_label)

            # Create a widget to contain the image and labels
            image_widget = QWidget(self)
            image_widget.setLayout(image_layout)

            # Add the widget to the grid layout
            row = idx // 2  # Every two images will be placed in the same row
            col = idx % 2  # Place images in the first or second column
            self.images_layout.addWidget(image_widget, row, col, 1, 1, Qt.AlignCenter)

            # Add spacing (adjust as needed)
            self.images_layout.setHorizontalSpacing(10)
            self.images_layout.setVerticalSpacing(10)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CannyEdgeApp()
    window.show()
    sys.exit(app.exec_())
