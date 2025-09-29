"""
Core Feature Extraction Methods Visualization
Generates figures demonstrating edge detection operators and basic feature extraction techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import feature, filters, morphology
from skimage.data import camera, coins, checkerboard
import matplotlib.patches as patches

# Set matplotlib style for consistency
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_gradient_operators():
    """Generate visualization of different gradient operators."""
    # Load a test image
    image = camera()

    # Define gradient operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    # Apply operators
    sobel_x_result = ndimage.convolve(image.astype(float), sobel_x)
    sobel_y_result = ndimage.convolve(image.astype(float), sobel_y)
    sobel_magnitude = np.sqrt(sobel_x_result**2 + sobel_y_result**2)

    prewitt_x_result = ndimage.convolve(image.astype(float), prewitt_x)
    prewitt_y_result = ndimage.convolve(image.astype(float), prewitt_y)
    prewitt_magnitude = np.sqrt(prewitt_x_result**2 + prewitt_y_result**2)

    roberts_x_result = ndimage.convolve(image.astype(float), roberts_x)
    roberts_y_result = ndimage.convolve(image.astype(float), roberts_y)
    roberts_magnitude = np.sqrt(roberts_x_result**2 + roberts_y_result**2)

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Sobel results
    axes[0, 1].imshow(sobel_magnitude, cmap='gray')
    axes[0, 1].set_title('Sobel Edge Detection')
    axes[0, 1].axis('off')

    # Prewitt results
    axes[0, 2].imshow(prewitt_magnitude, cmap='gray')
    axes[0, 2].set_title('Prewitt Edge Detection')
    axes[0, 2].axis('off')

    # Roberts results
    axes[0, 3].imshow(roberts_magnitude, cmap='gray')
    axes[0, 3].set_title('Roberts Edge Detection')
    axes[0, 3].axis('off')

    # Show kernels
    kernels = [sobel_x, prewitt_x, roberts_x, sobel_y]
    titles = ['Sobel X', 'Prewitt X', 'Roberts X', 'Sobel Y']

    for i, (kernel, title) in enumerate(zip(kernels, titles)):
        im = axes[1, i].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
        axes[1, i].set_title(f'{title} Kernel')
        axes[1, i].axis('off')

        # Add kernel values as text
        for (j, k), val in np.ndenumerate(kernel):
            axes[1, i].text(k, j, f'{val}', ha='center', va='center',
                           color='white' if abs(val) > 1 else 'black', fontweight='bold')

    plt.suptitle('Gradient Operators for Edge Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/gradient_operators.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_comparison():
    """Compare different edge detection methods."""
    image = camera()

    # Apply different edge detection methods
    sobel_edges = filters.sobel(image)
    canny_edges = feature.canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    laplacian_edges = filters.laplace(image)
    roberts_edges = filters.roberts(image)

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sobel_edges, cmap='gray')
    axes[0, 1].set_title('Sobel Edge Detection')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(canny_edges, cmap='gray')
    axes[0, 2].set_title('Canny Edge Detection')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(np.abs(laplacian_edges), cmap='gray')
    axes[1, 0].set_title('Laplacian Edge Detection')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(roberts_edges, cmap='gray')
    axes[1, 1].set_title('Roberts Edge Detection')
    axes[1, 1].axis('off')

    # Edge statistics comparison
    methods = ['Sobel', 'Canny', 'Laplacian', 'Roberts']
    edge_counts = [
        np.sum(sobel_edges > 0.1),
        np.sum(canny_edges),
        np.sum(np.abs(laplacian_edges) > 0.1),
        np.sum(roberts_edges > 0.1)
    ]

    axes[1, 2].bar(methods, edge_counts, color=['blue', 'green', 'red', 'orange'])
    axes[1, 2].set_title('Edge Pixel Count Comparison')
    axes[1, 2].set_ylabel('Number of Edge Pixels')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.suptitle('Edge Detection Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/edge_detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hough_transform_demo():
    """Demonstrate Hough transform for line and circle detection."""
    # Create synthetic image with lines and circles
    image = np.zeros((200, 200), dtype=np.uint8)

    # Add some lines
    cv2.line(image, (50, 50), (150, 100), 255, 2)
    cv2.line(image, (100, 30), (120, 170), 255, 2)
    cv2.line(image, (20, 150), (180, 160), 255, 2)

    # Add some circles
    cv2.circle(image, (60, 140), 25, 255, 2)
    cv2.circle(image, (140, 60), 20, 255, 2)

    # Add noise
    noise = np.random.normal(0, 20, image.shape).astype(np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)

    # Apply edge detection
    edges = feature.canny(image, sigma=1, low_threshold=50, high_threshold=150)

    # Hough line transform
    lines = cv2.HoughLines(edges.astype(np.uint8), 1, np.pi/180, threshold=50)

    # Hough circle transform
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                              param1=50, param2=30, minRadius=15, maxRadius=35)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection (Canny)')
    axes[0, 1].axis('off')

    # Hough lines
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), 128, 1)

    axes[1, 0].imshow(line_image, cmap='gray')
    axes[1, 0].set_title('Hough Line Detection')
    axes[1, 0].axis('off')

    # Hough circles
    circle_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circle_image, (x, y), 2, (0, 0, 255), 3)

    axes[1, 1].imshow(circle_image)
    axes[1, 1].set_title('Hough Circle Detection')
    axes[1, 1].axis('off')

    plt.suptitle('Hough Transform for Line and Circle Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/hough_transform_demo.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_texture_analysis():
    """Demonstrate basic texture analysis techniques."""
    # Create different texture patterns
    size = 128

    # Checkerboard pattern
    checkerboard_pattern = np.kron([[1, 0] * 8, [0, 1] * 8] * 8, np.ones((8, 8)))

    # Sine wave pattern
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    sine_pattern = np.sin(X) * np.cos(Y)

    # Random noise pattern
    noise_pattern = np.random.random((size, size))

    # Gabor-like pattern
    sigma_x, sigma_y = 20, 20
    theta = np.pi / 4
    freq = 0.1
    gabor_pattern = np.exp(-((X-2*np.pi)**2/(2*sigma_x**2) + (Y-2*np.pi)**2/(2*sigma_y**2)))
    gabor_pattern *= np.cos(freq * (X*np.cos(theta) + Y*np.sin(theta)))

    patterns = [checkerboard_pattern, sine_pattern, noise_pattern, gabor_pattern]
    pattern_names = ['Checkerboard', 'Sine Wave', 'Random Noise', 'Gabor-like']

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Show original patterns
    for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        axes[0, i].imshow(pattern, cmap='gray')
        axes[0, i].set_title(f'{name} Pattern')
        axes[0, i].axis('off')

    # Apply Gabor filters
    for i, pattern in enumerate(patterns):
        # Apply Gabor filter
        gabor_real, _ = filters.gabor(pattern, frequency=0.1, theta=0)
        axes[1, i].imshow(gabor_real, cmap='gray')
        axes[1, i].set_title(f'Gabor Filter (θ=0°)')
        axes[1, i].axis('off')

    # Local Binary Pattern
    for i, pattern in enumerate(patterns):
        # Normalize pattern to 0-255 range
        pattern_norm = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(pattern_norm, P=8, R=1, method='uniform')
        axes[2, i].imshow(lbp, cmap='gray')
        axes[2, i].set_title('Local Binary Pattern')
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Original\nPatterns', transform=axes[0, 0].transAxes,
                   rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Gabor\nFiltered', transform=axes[1, 0].transAxes,
                   rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    axes[2, 0].text(-0.1, 0.5, 'Local Binary\nPattern', transform=axes[2, 0].transAxes,
                   rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('Texture Analysis Techniques', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/texture_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_corner_detection():
    """Demonstrate corner detection algorithms."""
    # Create a synthetic image with corners
    image = np.zeros((200, 200), dtype=np.float32)

    # Add some rectangles and shapes with corners
    cv2.rectangle(image, (50, 50), (100, 100), 1.0, -1)
    cv2.rectangle(image, (120, 80), (170, 130), 1.0, -1)

    # Add a triangle
    triangle = np.array([[30, 150], [80, 150], [55, 180]], np.int32)
    cv2.fillPoly(image, [triangle], 1.0)

    # Add noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)

    # Harris corner detection
    harris_response = feature.corner_harris(image, sigma=1)
    harris_coords = feature.corner_peaks(harris_response, min_distance=10, threshold_rel=0.1)

    # Shi-Tomasi corner detection
    shi_tomasi_response = feature.corner_shi_tomasi(image, sigma=1)
    shi_tomasi_coords = feature.corner_peaks(shi_tomasi_response, min_distance=10, threshold_rel=0.1)

    # FAST corner detection
    fast_coords = feature.corner_fast(image, n=12, threshold=0.1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Harris corners
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].plot(harris_coords[:, 1], harris_coords[:, 0], 'r+', markersize=10, markeredgewidth=2)
    axes[0, 1].set_title(f'Harris Corners ({len(harris_coords)} detected)')
    axes[0, 1].axis('off')

    # Shi-Tomasi corners
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].plot(shi_tomasi_coords[:, 1], shi_tomasi_coords[:, 0], 'g+', markersize=10, markeredgewidth=2)
    axes[1, 0].set_title(f'Shi-Tomasi Corners ({len(shi_tomasi_coords)} detected)')
    axes[1, 0].axis('off')

    # FAST corners
    axes[1, 1].imshow(image, cmap='gray')
    if len(fast_coords) > 0:
        axes[1, 1].plot(fast_coords[:, 1], fast_coords[:, 0], 'b+', markersize=10, markeredgewidth=2)
    axes[1, 1].set_title(f'FAST Corners ({len(fast_coords)} detected)')
    axes[1, 1].axis('off')

    plt.suptitle('Corner Detection Algorithms', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/corner_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating core feature extraction visualizations...")

    print("Creating gradient operators visualization...")
    create_gradient_operators()

    print("Creating edge detection comparison...")
    create_edge_comparison()

    print("Creating Hough transform demonstration...")
    create_hough_transform_demo()

    print("Creating texture analysis visualization...")
    create_texture_analysis()

    print("Creating corner detection demonstration...")
    create_corner_detection()

    print("Core methods figures generated successfully!")