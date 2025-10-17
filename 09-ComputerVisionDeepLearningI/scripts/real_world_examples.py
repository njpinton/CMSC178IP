"""
Real-world examples for Computer Vision and Deep Learning I.
Generates visualizations using actual images and practical applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Get figures directory
figures_dir = Path(__file__).parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)

def create_sample_image():
    """Create a sample image for demonstrations."""
    # Create a simple synthetic image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add colored shapes
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)  # Blue square
    cv2.circle(img, (150, 50), 30, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (50, 120), (150, 180), (0, 0, 255), -1)  # Red rectangle

    return img

def generate_image_preprocessing():
    """Visualize image preprocessing steps."""
    # Create sample image
    img = create_sample_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Different preprocessing steps
    img_normalized = img_gray / 255.0
    img_equalized = cv2.equalizeHist(img_gray)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Original color
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Grayscale
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Conversion', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Normalized
    axes[0, 2].imshow(img_normalized, cmap='gray')
    axes[0, 2].set_title('Normalized [0, 1]', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Histogram equalized
    axes[1, 0].imshow(img_equalized, cmap='gray')
    axes[1, 0].set_title('Histogram Equalization', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Blurred
    axes[1, 1].imshow(img_blurred, cmap='gray')
    axes[1, 1].set_title('Gaussian Blur', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title('Edge Detection (Canny)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Image Preprocessing Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "11_preprocessing_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 11_preprocessing_pipeline.png")

def generate_data_augmentation():
    """Visualize data augmentation techniques."""
    # Create sample image
    img = create_sample_image()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # Horizontal flip
    img_hflip = cv2.flip(img_rgb, 1)
    axes[0, 1].imshow(img_hflip)
    axes[0, 1].set_title('Horizontal Flip', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # Vertical flip
    img_vflip = cv2.flip(img_rgb, 0)
    axes[0, 2].imshow(img_vflip)
    axes[0, 2].set_title('Vertical Flip', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    # Rotation
    rows, cols = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    img_rotated = cv2.warpAffine(img_rgb, M, (cols, rows))
    axes[0, 3].imshow(img_rotated)
    axes[0, 3].set_title('Rotation (45°)', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')

    # Brightness increase
    img_bright = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=30)
    axes[1, 0].imshow(img_bright)
    axes[1, 0].set_title('Brightness +', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Brightness decrease
    img_dark = cv2.convertScaleAbs(img_rgb, alpha=0.6, beta=-20)
    axes[1, 1].imshow(img_dark)
    axes[1, 1].set_title('Brightness -', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    # Scaling (zoom)
    img_scaled = cv2.resize(img_rgb[50:150, 50:150], (200, 200))
    axes[1, 2].imshow(img_scaled)
    axes[1, 2].set_title('Zoom/Crop', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    # Add noise
    noise = np.random.randn(*img_rgb.shape) * 25
    img_noisy = np.clip(img_rgb + noise, 0, 255).astype(np.uint8)
    axes[1, 3].imshow(img_noisy)
    axes[1, 3].set_title('Gaussian Noise', fontsize=11, fontweight='bold')
    axes[1, 3].axis('off')

    plt.suptitle('Data Augmentation Techniques', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "12_data_augmentation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 12_data_augmentation.png")

def generate_classification_example():
    """Generate a classification problem visualization."""
    np.random.seed(42)

    # Generate synthetic 2D classification data
    n_samples = 100
    class1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    class2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    class3 = np.random.randn(n_samples, 2) + np.array([2, -2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    axes[0].scatter(class1[:, 0], class1[:, 1], c='#2E86AB', label='Class 1: Cat',
                   s=50, alpha=0.6, edgecolors='black')
    axes[0].scatter(class2[:, 0], class2[:, 1], c='#A23B72', label='Class 2: Dog',
                   s=50, alpha=0.6, edgecolors='black')
    axes[0].scatter(class3[:, 0], class3[:, 1], c='#F18F01', label='Class 3: Bird',
                   s=50, alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].set_title('Feature Space Representation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Decision boundaries (simplified)
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))

    # Simple decision regions
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            d1 = np.linalg.norm(point - np.array([2, 2]))
            d2 = np.linalg.norm(point - np.array([-2, -2]))
            d3 = np.linalg.norm(point - np.array([2, -2]))
            Z[i, j] = np.argmin([d1, d2, d3])

    axes[1].contourf(xx, yy, Z, alpha=0.3, levels=2, colors=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1].scatter(class1[:, 0], class1[:, 1], c='#2E86AB', label='Class 1: Cat',
                   s=50, alpha=0.6, edgecolors='black')
    axes[1].scatter(class2[:, 0], class2[:, 1], c='#A23B72', label='Class 2: Dog',
                   s=50, alpha=0.6, edgecolors='black')
    axes[1].scatter(class3[:, 0], class3[:, 1], c='#F18F01', label='Class 3: Bird',
                   s=50, alpha=0.6, edgecolors='black')
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].set_title('Decision Boundaries', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Multi-Class Classification Problem', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_classification_example.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 13_classification_example.png")

def generate_confusion_matrix():
    """Generate confusion matrix visualization."""
    # Synthetic confusion matrix
    confusion_matrix = np.array([
        [85, 8, 7],
        [12, 78, 10],
        [5, 15, 80]
    ])

    classes = ['Cat', 'Dog', 'Bird']

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12)

    # Set ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="white" if confusion_matrix[i, j] > 50 else "black",
                          fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Multi-Class Classification', fontsize=16, fontweight='bold', pad=20)

    # Calculate and display metrics
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
           ha='center', transform=ax.transAxes, fontsize=13,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(figures_dir / "14_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 14_confusion_matrix.png")

def generate_overfitting_example():
    """Visualize overfitting vs good fit."""
    np.random.seed(42)

    # Generate data
    x = np.linspace(0, 10, 50)
    y_true = 2 * np.sin(x) + x
    y_noisy = y_true + np.random.randn(50) * 1.5

    # Fits
    x_dense = np.linspace(0, 10, 200)
    y_good = 2 * np.sin(x_dense) + x_dense
    y_overfit = y_good + np.sin(x_dense * 5) * 0.8

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Good fit
    axes[0].scatter(x, y_noisy, alpha=0.6, s=50, color='gray', label='Training Data')
    axes[0].plot(x_dense, y_good, linewidth=3, color='#2E86AB', label='Model Prediction')
    axes[0].set_xlabel('Input Feature', fontsize=12)
    axes[0].set_ylabel('Output', fontsize=12)
    axes[0].set_title('Good Fit (Generalizes Well)', fontsize=14, fontweight='bold', color='green')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Overfitting
    axes[1].scatter(x, y_noisy, alpha=0.6, s=50, color='gray', label='Training Data')
    axes[1].plot(x_dense, y_overfit, linewidth=3, color='#C73E1D', label='Model Prediction')
    axes[1].set_xlabel('Input Feature', fontsize=12)
    axes[1].set_ylabel('Output', fontsize=12)
    axes[1].set_title('Overfitting (Memorizes Noise)', fontsize=14, fontweight='bold', color='red')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Overfitting vs Good Generalization', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "15_overfitting_example.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 15_overfitting_example.png")

def main():
    """Generate all real-world example figures."""
    print("Generating real-world example figures...")
    print("-" * 40)

    generate_image_preprocessing()
    generate_data_augmentation()
    generate_classification_example()
    generate_confusion_matrix()
    generate_overfitting_example()

    print("-" * 40)
    print("Real-world examples figures complete!")

if __name__ == "__main__":
    main()
