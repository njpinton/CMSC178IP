#!/usr/bin/env python3
"""
Generate JPEG compression procedure demonstration using sklearn dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import cv2
from PIL import Image
import io
import os
# Use OpenCV for DCT operations instead of scipy

# Create images folder
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Set matplotlib parameters
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

def save_figure(filename):
    """Save figure with consistent formatting"""
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def create_sklearn_jpeg_procedure():
    """Create JPEG compression procedure using sklearn digits dataset"""

    # Load sklearn digits dataset (8x8 images)
    digits = load_digits()

    # Select a clear digit (let's use digit '5' which has good contrast)
    digit_5_indices = np.where(digits.target == 5)[0]
    sample_image = digits.images[digit_5_indices[0]]  # First instance of digit 5

    # Resize to 64x64 for better visualization (8x8 is too small)
    sample_resized = cv2.resize(sample_image, (64, 64), interpolation=cv2.INTER_NEAREST)

    # Create figure with 6 subplots showing the JPEG process
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('JPEG Compression Process using sklearn Digits Dataset',
                 fontsize=20, fontweight='bold', y=0.95)

    # Step 1: Original Image
    ax1 = axes[0, 0]
    im1 = ax1.imshow(sample_resized, cmap='gray', interpolation='nearest')
    ax1.set_title('1. Original Image\n(sklearn digits: "5")', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Add colorbar for reference
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Pixel Intensity', fontsize=10)

    # Step 2: Color Space Conversion (RGB to YUV)
    # Since our image is grayscale, we'll simulate the Y channel
    ax2 = axes[0, 1]
    # For demonstration, show the luminance component
    y_component = sample_resized  # Already grayscale, so Y = grayscale
    im2 = ax2.imshow(y_component, cmap='gray', interpolation='nearest')
    ax2.set_title('2. Luminance Component (Y)\n(Grayscale = Y channel)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Y Value', fontsize=10)

    # Step 3: 8x8 Block Division and DCT
    ax3 = axes[0, 2]
    # Show 8x8 block grid overlay
    block_image = sample_resized.copy()

    # Draw 8x8 grid lines
    for i in range(0, 64, 8):
        block_image[i, :] = 255  # Horizontal lines
        block_image[:, i] = 255  # Vertical lines

    im3 = ax3.imshow(block_image, cmap='gray', interpolation='nearest')
    ax3.set_title('3. 8×8 Block Division\n(Grid overlay shown)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Step 4: DCT Coefficients (show one 8x8 block)
    ax4 = axes[1, 0]
    # Extract one 8x8 block (top-left)
    block_8x8 = sample_resized[0:8, 0:8].astype(np.float32)

    # Apply DCT
    dct_coeffs = cv2.dct(block_8x8 - 128)  # Center around 0

    im4 = ax4.imshow(np.abs(dct_coeffs), cmap='hot', interpolation='nearest')
    ax4.set_title('4. DCT Coefficients\n(One 8×8 block)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('|DCT Coeff|', fontsize=10)

    # Add text annotations for DC and AC components
    ax4.text(0, -0.8, 'DC', fontsize=12, fontweight='bold', ha='center', color='white',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    ax4.text(7, 7, 'AC', fontsize=12, fontweight='bold', ha='center', color='black',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Step 5: Quantization
    ax5 = axes[1, 1]
    # Standard JPEG quantization matrix
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Apply quantization
    quantized = np.round(dct_coeffs / quant_matrix) * quant_matrix

    im5 = ax5.imshow(np.abs(quantized), cmap='hot', interpolation='nearest')
    ax5.set_title('5. After Quantization\n(Information loss occurs)', fontsize=14, fontweight='bold')
    ax5.axis('off')

    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('|Quantized|', fontsize=10)

    # Step 6: Reconstructed Image
    ax6 = axes[1, 2]

    # Reconstruct the full image with JPEG compression simulation
    reconstructed = np.zeros_like(sample_resized, dtype=np.float32)

    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            # Extract 8x8 block
            block = sample_resized[i:i+8, j:j+8].astype(np.float32)

            # DCT
            dct_block = cv2.dct(block - 128)

            # Quantize
            quantized_block = np.round(dct_block / quant_matrix) * quant_matrix

            # Inverse DCT
            reconstructed_block = cv2.idct(quantized_block) + 128

            # Clip values
            reconstructed_block = np.clip(reconstructed_block, 0, 255)

            # Place back
            reconstructed[i:i+8, j:j+8] = reconstructed_block

    im6 = ax6.imshow(reconstructed, cmap='gray', interpolation='nearest')
    ax6.set_title('6. Reconstructed Image\n(After JPEG compression)', fontsize=14, fontweight='bold')
    ax6.axis('off')

    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Pixel Intensity', fontsize=10)

    # Calculate and display PSNR
    mse = np.mean((sample_resized - reconstructed) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    # Add quality metrics
    fig.text(0.5, 0.02, f'Quality Metrics: PSNR = {psnr:.2f} dB | MSE = {mse:.2f}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Add process arrows between subplots
    # This is done with annotations
    fig.text(0.22, 0.75, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.55, 0.75, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.22, 0.25, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.55, 0.25, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.88, 0.5, '↓', fontsize=30, fontweight='bold', ha='center', va='center', rotation=90)

    save_figure('jpeg_procedure_steps.png')

def create_sklearn_comparison():
    """Create a comparison showing different sklearn dataset samples"""

    # Load different datasets for variety
    digits = load_digits()

    plt.figure(figsize=(16, 10))

    # Show multiple digits with JPEG compression applied
    for i in range(10):  # Show digits 0-9
        digit_indices = np.where(digits.target == i)[0]
        sample_image = digits.images[digit_indices[0]]

        # Original
        plt.subplot(4, 10, i+1)
        plt.imshow(sample_image, cmap='gray', interpolation='nearest')
        plt.title(f'Original\nDigit {i}', fontsize=10, fontweight='bold')
        plt.axis('off')

        # Resize for JPEG simulation
        sample_64 = cv2.resize(sample_image, (64, 64), interpolation=cv2.INTER_NEAREST)

        # High quality JPEG (Q=90)
        compressed_high = simulate_jpeg_compression(sample_64, quality=90)
        plt.subplot(4, 10, i+11)
        plt.imshow(compressed_high, cmap='gray', interpolation='nearest')
        plt.title('JPEG Q=90\n(High)', fontsize=10, fontweight='bold')
        plt.axis('off')

        # Medium quality JPEG (Q=50)
        compressed_med = simulate_jpeg_compression(sample_64, quality=50)
        plt.subplot(4, 10, i+21)
        plt.imshow(compressed_med, cmap='gray', interpolation='nearest')
        plt.title('JPEG Q=50\n(Medium)', fontsize=10, fontweight='bold')
        plt.axis('off')

        # Low quality JPEG (Q=10)
        compressed_low = simulate_jpeg_compression(sample_64, quality=10)
        plt.subplot(4, 10, i+31)
        plt.imshow(compressed_low, cmap='gray', interpolation='nearest')
        plt.title('JPEG Q=10\n(Low)', fontsize=10, fontweight='bold')
        plt.axis('off')

    plt.suptitle('JPEG Compression Quality Comparison - sklearn Digits Dataset',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure('sklearn_jpeg_quality_comparison.png')

def simulate_jpeg_compression(image, quality):
    """Simulate JPEG compression with different quality levels"""

    # Quality to quantization scale mapping
    if quality >= 50:
        scale = (100 - quality) / 50.0
    else:
        scale = 50.0 / quality

    # Base quantization matrix
    base_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Scale quantization matrix based on quality
    quant_matrix = base_quant * scale
    quant_matrix = np.clip(quant_matrix, 1, 255)

    # Ensure image is the right size (multiple of 8)
    h, w = image.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
    else:
        padded = image

    compressed = np.zeros_like(padded, dtype=np.float32)

    # Process each 8x8 block
    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block = padded[i:i+8, j:j+8].astype(np.float32)

            # DCT
            dct_block = cv2.dct(block - 128)

            # Quantize
            quantized = np.round(dct_block / quant_matrix) * quant_matrix

            # Inverse DCT
            reconstructed = cv2.idct(quantized) + 128
            reconstructed = np.clip(reconstructed, 0, 255)

            compressed[i:i+8, j:j+8] = reconstructed

    # Remove padding if added
    if pad_h > 0 or pad_w > 0:
        compressed = compressed[:h, :w]

    return compressed

if __name__ == "__main__":
    print("Generating sklearn-based JPEG procedure demonstration...")

    print("1. Creating JPEG procedure steps with sklearn digits...")
    create_sklearn_jpeg_procedure()

    print("2. Creating quality comparison...")
    create_sklearn_comparison()

    print("sklearn JPEG demonstrations generated successfully!")