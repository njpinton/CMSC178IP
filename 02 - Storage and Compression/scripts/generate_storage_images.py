#!/usr/bin/env python3
"""
Generate sample images for Storage and Compression presentation
Creates examples of different image types, storage formats, and compression techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from scipy import ndimage
from skimage import data, filters, restoration, transform
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import struct

# Create images folder
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Set matplotlib parameters for consistent output
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

def save_figure(filename, figsize=(8, 6)):
    """Save figure with consistent formatting"""
    plt.tight_layout()
    plt.savefig(f"{images_dir}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def create_sample_base_image():
    """Create a sample image for demonstrations"""
    # Create a colorful test image with various features
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Add gradient background
    for i in range(256):
        for j in range(256):
            img[i, j, 0] = int(255 * (i / 255))  # Red gradient
            img[i, j, 1] = int(255 * (j / 255))  # Green gradient
            img[i, j, 2] = int(255 * ((i + j) / 510))  # Blue gradient

    # Add some geometric shapes
    cv2.circle(img, (64, 64), 30, (255, 255, 255), -1)
    cv2.rectangle(img, (150, 150), (200, 200), (0, 0, 0), -1)
    cv2.circle(img, (200, 64), 25, (255, 0, 0), 3)

    return img

def generate_image_types():
    """Generate examples of different image types"""
    base_img = create_sample_base_image()

    # 1. Binary Image
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image\n(1-bit per pixel)', fontsize=10)
    plt.axis('off')

    # 2. Grayscale Image
    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image\n(8-bit per pixel)', fontsize=10)
    plt.axis('off')

    # 3. Color Image (RGB)
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    plt.title('Color Image (RGB)\n(24-bit per pixel)', fontsize=10)
    plt.axis('off')

    # 4. Indexed Color Image
    # Create an indexed color version
    pil_img = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    indexed_img = pil_img.quantize(colors=16)
    indexed_array = np.array(indexed_img)

    plt.subplot(1, 4, 4)
    plt.imshow(indexed_array, cmap='tab20')
    plt.title('Indexed Color\n(4-bit per pixel)', fontsize=10)
    plt.axis('off')

    save_figure('image_types_comparison.png', figsize=(12, 3))

def generate_storage_formats():
    """Generate visual comparison of storage formats"""
    base_img = create_sample_base_image()

    # Save in different formats and show file sizes
    formats = []

    # BMP (uncompressed)
    cv2.imwrite(f"{images_dir}/temp_bmp.bmp", base_img)
    bmp_size = os.path.getsize(f"{images_dir}/temp_bmp.bmp")
    formats.append(("BMP", bmp_size, "Uncompressed"))

    # PNG (lossless compression)
    cv2.imwrite(f"{images_dir}/temp_png.png", base_img)
    png_size = os.path.getsize(f"{images_dir}/temp_png.png")
    formats.append(("PNG", png_size, "Lossless"))

    # JPEG (lossy compression)
    cv2.imwrite(f"{images_dir}/temp_jpg.jpg", base_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpg_size = os.path.getsize(f"{images_dir}/temp_jpg.jpg")
    formats.append(("JPEG", jpg_size, "Lossy"))

    # Create comparison chart
    plt.figure(figsize=(10, 6))

    format_names = [f[0] for f in formats]
    file_sizes = [f[1]/1024 for f in formats]  # Convert to KB
    compression_types = [f[2] for f in formats]

    colors = ['red', 'blue', 'green']
    bars = plt.bar(format_names, file_sizes, color=colors, alpha=0.7)

    plt.title('File Size Comparison by Format', fontsize=14, fontweight='bold')
    plt.ylabel('File Size (KB)', fontsize=12)
    plt.xlabel('Format', fontsize=12)

    # Add labels on bars
    for i, (bar, comp_type) in enumerate(zip(bars, compression_types)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f} KB\n({comp_type})',
                ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', alpha=0.3)
    save_figure('storage_formats_comparison.png', figsize=(10, 6))

    # Clean up temp files
    for ext in ['bmp', 'png', 'jpg']:
        if os.path.exists(f"{images_dir}/temp_{ext}.{ext}"):
            os.remove(f"{images_dir}/temp_{ext}.{ext}")

def generate_compression_demonstration():
    """Generate visual demonstration of compression effects"""
    base_img = create_sample_base_image()

    # Create JPEG compression at different quality levels
    qualities = [10, 30, 60, 90]

    plt.figure(figsize=(16, 4))

    for i, quality in enumerate(qualities):
        # Compress and decompress
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', base_img, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        # Save temporary file to get file size
        temp_filename = f"{images_dir}/temp_q{quality}.jpg"
        cv2.imwrite(temp_filename, base_img, encode_param)
        file_size = os.path.getsize(temp_filename)
        os.remove(temp_filename)

        plt.subplot(1, 4, i+1)
        plt.imshow(cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB))
        plt.title(f'Quality {quality}%\nSize: {file_size/1024:.1f} KB', fontsize=10)
        plt.axis('off')

    save_figure('jpeg_compression_quality.png', figsize=(16, 4))

def generate_compression_artifacts():
    """Generate examples of compression artifacts"""
    # Create a high-detail test image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add fine details that show compression artifacts
    for i in range(0, 200, 2):
        cv2.line(img, (i, 0), (i, 200), (255, 255, 255), 1)
    for i in range(0, 200, 2):
        cv2.line(img, (0, i), (200, i), (255, 255, 255), 1)

    # Add some text
    cv2.putText(img, 'DETAIL', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12)
    plt.axis('off')

    # Heavy JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    jpeg_img = cv2.imdecode(encimg, 1)

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(jpeg_img, cv2.COLOR_BGR2RGB))
    plt.title('JPEG Compression\n(Quality 10%)', fontsize=12)
    plt.axis('off')

    # PNG (lossless)
    _, encimg = cv2.imencode('.png', img)
    png_img = cv2.imdecode(encimg, 1)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(png_img, cv2.COLOR_BGR2RGB))
    plt.title('PNG Compression\n(Lossless)', fontsize=12)
    plt.axis('off')

    save_figure('compression_artifacts.png', figsize=(12, 4))

def generate_huffman_encoding_demo():
    """Generate visual demonstration of Huffman encoding"""
    # Sample text for Huffman encoding
    text = "ABRACADABRA"

    # Count frequencies
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # Create frequency chart
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    chars = list(freq.keys())
    frequencies = list(freq.values())
    plt.bar(chars, frequencies, color='steelblue')
    plt.title('Character Frequencies', fontsize=12)
    plt.xlabel('Characters')
    plt.ylabel('Frequency')

    # Show original vs compressed representation
    plt.subplot(2, 2, 2)
    original_bits = len(text) * 8  # ASCII encoding

    # Simplified Huffman encoding (theoretical)
    huffman_codes = {'A': '0', 'B': '10', 'R': '110', 'C': '1110', 'D': '1111'}
    compressed_bits = sum(len(huffman_codes[char]) * freq[char] for char in freq)

    methods = ['Original\n(ASCII)', 'Huffman\nEncoded']
    bits = [original_bits, compressed_bits]
    colors = ['red', 'green']

    bars = plt.bar(methods, bits, color=colors, alpha=0.7)
    plt.title('Compression Comparison', fontsize=12)
    plt.ylabel('Total Bits')

    for bar, bit_count in zip(bars, bits):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bit_count} bits',
                ha='center', va='bottom')

    # Show Huffman tree (simplified)
    plt.subplot(2, 1, 2)
    plt.text(0.1, 0.8, 'Huffman Codes:', fontsize=14, fontweight='bold')
    y_pos = 0.6
    for char, code in huffman_codes.items():
        plt.text(0.1, y_pos, f'{char}: {code}', fontsize=12)
        y_pos -= 0.1

    compression_ratio = (1 - compressed_bits/original_bits) * 100
    plt.text(0.5, 0.3, f'Compression Ratio: {compression_ratio:.1f}%',
             fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    save_figure('huffman_encoding_demo.png', figsize=(12, 8))

def generate_run_length_encoding_demo():
    """Generate visual demonstration of Run Length Encoding"""
    # Create a simple pattern with runs
    pattern = "AAABBBBCCAAA"

    plt.figure(figsize=(12, 6))

    # Original data visualization
    plt.subplot(2, 1, 1)
    chars = list(pattern)
    x_pos = range(len(chars))
    colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
    bar_colors = [colors[char] for char in chars]

    plt.bar(x_pos, [1]*len(chars), color=bar_colors)
    plt.title('Original Data: ' + pattern, fontsize=14)
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.xticks(x_pos, chars)

    # RLE encoded visualization
    plt.subplot(2, 1, 2)
    rle_encoded = "3A4B2C3A"

    plt.text(0.1, 0.7, 'Run Length Encoded:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.5, f'Original: {pattern} ({len(pattern)} characters)', fontsize=12)
    plt.text(0.1, 0.3, f'RLE: {rle_encoded} ({len(rle_encoded)} characters)', fontsize=12)

    compression_ratio = (1 - len(rle_encoded)/len(pattern)) * 100
    plt.text(0.1, 0.1, f'Compression Ratio: {compression_ratio:.1f}%',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    save_figure('rle_encoding_demo.png', figsize=(12, 6))

def generate_dct_demo():
    """Generate visual demonstration of DCT (Discrete Cosine Transform)"""
    # Create an 8x8 block
    block = np.random.randint(0, 256, (8, 8)).astype(np.float32)

    # Apply DCT
    dct_block = cv2.dct(block)

    # Apply quantization (JPEG-style)
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    quantized_dct = np.round(dct_block / quantization_matrix) * quantization_matrix

    # Inverse DCT
    reconstructed = cv2.idct(quantized_dct)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(block, cmap='gray')
    plt.title('Original 8×8 Block', fontsize=10)
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(np.abs(dct_block), cmap='hot')
    plt.title('DCT Coefficients', fontsize=10)
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(np.abs(quantized_dct), cmap='hot')
    plt.title('Quantized DCT', fontsize=10)
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Block', fontsize=10)
    plt.colorbar()

    save_figure('dct_demonstration.png', figsize=(15, 5))

def generate_bit_depth_comparison():
    """Generate comparison of different bit depths"""
    base_img = cv2.cvtColor(create_sample_base_image(), cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(16, 4))

    bit_depths = [1, 2, 4, 8]

    for i, bits in enumerate(bit_depths):
        levels = 2 ** bits
        quantized = np.round(base_img / 256 * levels) * (256 / levels)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        plt.subplot(1, 4, i+1)
        plt.imshow(quantized, cmap='gray')
        plt.title(f'{bits}-bit\n({levels} levels)', fontsize=12)
        plt.axis('off')

    save_figure('bit_depth_comparison.png', figsize=(16, 4))

def generate_color_spaces_demo():
    """Generate demonstration of different color spaces"""
    base_img = create_sample_base_image()
    rgb_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))

    # RGB
    plt.subplot(2, 4, 1)
    plt.imshow(rgb_img)
    plt.title('RGB', fontsize=12)
    plt.axis('off')

    # Individual RGB channels
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(2, 4, i+2)
        channel = np.zeros_like(rgb_img)
        channel[:, :, i] = rgb_img[:, :, i]
        plt.imshow(channel)
        plt.title(f'{color} Channel', fontsize=12)
        plt.axis('off')

    # HSV
    hsv_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    plt.title('HSV', fontsize=12)
    plt.axis('off')

    # Individual HSV channels
    for i, component in enumerate(['Hue', 'Saturation', 'Value']):
        plt.subplot(2, 4, i+6)
        plt.imshow(hsv_img[:, :, i], cmap='gray')
        plt.title(f'{component}', fontsize=12)
        plt.axis('off')

    save_figure('color_spaces_demo.png', figsize=(16, 8))

def main():
    """Generate all sample images"""
    print("Generating storage and compression demonstration images...")

    print("1. Image types comparison...")
    generate_image_types()

    print("2. Storage formats comparison...")
    generate_storage_formats()

    print("3. Compression quality demonstration...")
    generate_compression_demonstration()

    print("4. Compression artifacts...")
    generate_compression_artifacts()

    print("5. Huffman encoding demo...")
    generate_huffman_encoding_demo()

    print("6. Run Length Encoding demo...")
    generate_run_length_encoding_demo()

    print("7. DCT demonstration...")
    generate_dct_demo()

    print("8. Bit depth comparison...")
    generate_bit_depth_comparison()

    print("9. Color spaces demonstration...")
    generate_color_spaces_demo()

    print(f"All images generated successfully in '{images_dir}' directory!")
    print(f"Generated {len(os.listdir(images_dir))} images.")

if __name__ == "__main__":
    main()