#!/usr/bin/env python3
"""
Generate demonstration images using scikit-image datasets
Better suited for image processing examples than sklearn digits
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, transform, restoration, segmentation
from skimage.util import img_as_ubyte, img_as_float
import cv2
from PIL import Image
import os

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

def create_skimage_jpeg_procedure():
    """Create JPEG compression procedure using scikit-image datasets"""

    # Use the classic 'cameraman' image from scikit-image
    original_image = data.camera()  # 512x512 grayscale image

    # Resize to a manageable size for demonstration (256x256)
    resized_image = transform.resize(original_image, (256, 256),
                                   anti_aliasing=True, preserve_range=True).astype(np.uint8)

    # Create figure with 6 subplots showing the JPEG process
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('JPEG Compression Process using scikit-image "Camera" Dataset',
                 fontsize=20, fontweight='bold', y=0.95)

    # Step 1: Original Image
    ax1 = axes[0, 0]
    im1 = ax1.imshow(resized_image, cmap='gray', interpolation='nearest')
    ax1.set_title('1. Original Image\n(scikit-image camera)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Add colorbar for reference
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Pixel Intensity', fontsize=10)

    # Step 2: Color Space Conversion (already grayscale, so show Y component)
    ax2 = axes[0, 1]
    y_component = resized_image  # Already grayscale
    im2 = ax2.imshow(y_component, cmap='gray', interpolation='nearest')
    ax2.set_title('2. Luminance Component (Y)\n(Already grayscale)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Y Value', fontsize=10)

    # Step 3: 8x8 Block Division
    ax3 = axes[0, 2]
    # Show 8x8 block grid overlay on a cropped section for visibility
    block_demo = resized_image[64:128, 64:128].copy()  # 64x64 section for clarity

    # Draw 8x8 grid lines
    for i in range(0, 64, 8):
        block_demo[i, :] = 255  # Horizontal lines
        block_demo[:, i] = 255  # Vertical lines

    im3 = ax3.imshow(block_demo, cmap='gray', interpolation='nearest')
    ax3.set_title('3. 8×8 Block Division\n(64×64 section shown)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Step 4: DCT Coefficients (show one 8x8 block)
    ax4 = axes[1, 0]
    # Extract one 8x8 block from an interesting area
    block_8x8 = resized_image[100:108, 100:108].astype(np.float32)

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
    reconstructed = simulate_jpeg_compression(resized_image, quality=50)

    im6 = ax6.imshow(reconstructed, cmap='gray', interpolation='nearest')
    ax6.set_title('6. Reconstructed Image\n(After JPEG compression)', fontsize=14, fontweight='bold')
    ax6.axis('off')

    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Pixel Intensity', fontsize=10)

    # Calculate and display PSNR
    mse = np.mean((resized_image.astype(float) - reconstructed.astype(float)) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    # Add quality metrics
    fig.text(0.5, 0.02, f'Quality Metrics: PSNR = {psnr:.2f} dB | MSE = {mse:.2f}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Add process arrows between subplots
    fig.text(0.22, 0.75, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.55, 0.75, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.22, 0.25, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.55, 0.25, '→', fontsize=30, fontweight='bold', ha='center', va='center')
    fig.text(0.88, 0.5, '↓', fontsize=30, fontweight='bold', ha='center', va='center', rotation=90)

    save_figure('jpeg_procedure_steps.png')

def create_skimage_compression_artifacts():
    """Create compression artifacts demonstration using scikit-image"""

    # Use the Lena image for artifact demonstration
    original = data.astronaut()  # Color image

    # Convert to grayscale for JPEG demo
    gray_original = color.rgb2gray(original)
    gray_original = img_as_ubyte(gray_original)

    plt.figure(figsize=(16, 12))

    # Quality levels to demonstrate
    qualities = [90, 50, 20, 5]

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(gray_original, cmap='gray')
    plt.title('Original Image\n(scikit-image astronaut)', fontsize=14, fontweight='bold')
    plt.axis('off')

    # Different compression levels
    for i, quality in enumerate(qualities):
        compressed = simulate_jpeg_compression(gray_original, quality)

        plt.subplot(2, 3, i+2)
        plt.imshow(compressed, cmap='gray')

        # Calculate PSNR for this quality
        mse = np.mean((gray_original.astype(float) - compressed.astype(float)) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

        plt.title(f'JPEG Quality {quality}\nPSNR: {psnr:.2f} dB', fontsize=12, fontweight='bold')
        plt.axis('off')

    # Difference image (artifacts visualization)
    plt.subplot(2, 3, 6)
    # Show artifacts from quality 20
    compressed_low = simulate_jpeg_compression(gray_original, 20)
    diff = np.abs(gray_original.astype(float) - compressed_low.astype(float))

    plt.imshow(diff, cmap='hot')
    plt.title('Compression Artifacts\n(Difference from Original)', fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle('JPEG Compression Quality Comparison - scikit-image Dataset',
                 fontsize=16, fontweight='bold')
    save_figure('compression_artifacts.png')

def create_skimage_image_types():
    """Create image types demonstration using scikit-image datasets"""

    # Use coins image for good binary/grayscale examples
    coins = data.coins()

    plt.figure(figsize=(16, 8))

    # 1. Original grayscale
    plt.subplot(2, 4, 1)
    plt.imshow(coins, cmap='gray')
    plt.title('Grayscale Original\n(scikit-image coins)', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 2. Binary image (using Otsu threshold)
    from skimage.filters import threshold_otsu
    threshold = threshold_otsu(coins)
    binary = coins > threshold

    plt.subplot(2, 4, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image\n(Otsu threshold)', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 3. Reduced bit depth examples
    bit_depths = [4, 2, 1]
    for i, bits in enumerate(bit_depths):
        levels = 2 ** bits
        quantized = np.round(coins / 256 * levels) * (256 / levels)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        plt.subplot(2, 4, i+3)
        plt.imshow(quantized, cmap='gray')
        plt.title(f'{bits}-bit Quantized\n({levels} gray levels)', fontsize=12, fontweight='bold')
        plt.axis('off')

    # 4. Use color image for RGB demonstration
    astronaut = data.astronaut()

    plt.subplot(2, 4, 6)
    plt.imshow(astronaut)
    plt.title('Color Image (RGB)\n(scikit-image astronaut)', fontsize=12, fontweight='bold')
    plt.axis('off')

    # 5. Color channels separated
    plt.subplot(2, 4, 7)
    red_channel = astronaut[:, :, 0]
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel\nOnly', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    # Create indexed color version
    from sklearn.cluster import KMeans
    # Reshape image for clustering
    img_reshaped = astronaut.reshape(-1, 3)
    # Use KMeans to create palette
    kmeans = KMeans(n_clusters=16, random_state=42)
    kmeans.fit(img_reshaped)
    indexed = kmeans.cluster_centers_[kmeans.labels_].reshape(astronaut.shape)

    plt.imshow(indexed.astype(np.uint8))
    plt.title('Indexed Color\n(16 colors, K-means)', fontsize=12, fontweight='bold')
    plt.axis('off')

    save_figure('image_types_comparison.png')

def create_skimage_storage_formats():
    """Create storage format comparison using scikit-image"""

    # Use multiple scikit-image datasets for format comparison
    images_data = {
        'Camera': data.camera(),
        'Coins': data.coins(),
        'Checkerboard': data.checkerboard(),
        'Text': data.text()
    }

    plt.figure(figsize=(16, 12))

    format_info = []

    for i, (name, img) in enumerate(images_data.items()):
        # Original image
        plt.subplot(4, 5, i*5 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{name}\nOriginal', fontsize=10, fontweight='bold')
        plt.axis('off')

        # Simulate different storage formats by file size/compression
        formats = ['PNG', 'JPEG-90', 'JPEG-50', 'JPEG-10']
        qualities = [None, 90, 50, 10]  # PNG has no quality setting

        for j, (fmt, quality) in enumerate(zip(formats, qualities)):
            plt.subplot(4, 5, i*5 + j + 2)

            if fmt == 'PNG':
                # PNG is lossless, so show original
                processed = img
                size_factor = 0.8  # PNG typically smaller than raw but larger than JPEG
            else:
                # JPEG compression
                processed = simulate_jpeg_compression(img, quality)
                size_factor = 0.1 + (quality / 100) * 0.5  # Rough size estimation

            plt.imshow(processed, cmap='gray')

            # Calculate quality metric
            if fmt != 'PNG':
                mse = np.mean((img.astype(float) - processed.astype(float)) ** 2)
                psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
                plt.title(f'{fmt}\nPSNR: {psnr:.1f}dB', fontsize=9, fontweight='bold')
            else:
                plt.title(f'{fmt}\nLossless', fontsize=9, fontweight='bold')

            plt.axis('off')

    plt.suptitle('Storage Format Comparison - scikit-image Datasets',
                 fontsize=16, fontweight='bold')
    save_figure('storage_formats_comparison.png')

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
    quant_matrix = np.maximum(base_quant * scale, 1).astype(int)

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

    return compressed.astype(np.uint8)

if __name__ == "__main__":
    print("Generating scikit-image based demonstration images...")

    print("1. Creating JPEG procedure with scikit-image camera...")
    create_skimage_jpeg_procedure()

    print("2. Creating compression artifacts with scikit-image astronaut...")
    create_skimage_compression_artifacts()

    print("3. Creating image types demonstration...")
    create_skimage_image_types()

    print("4. Creating storage format comparison...")
    create_skimage_storage_formats()

    print("scikit-image demonstrations generated successfully!")