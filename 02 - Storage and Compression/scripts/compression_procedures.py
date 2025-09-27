#!/usr/bin/env python3
"""
Compression Procedures Demonstration
Step-by-step visual demonstration of compression algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

# Create images folder if it doesn't exist
images_dir = "images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

def generate_huffman_procedure():
    """Step-by-step Huffman encoding procedure"""
    # Sample data
    text = "ABRACADABRA"

    # Step 1: Frequency Analysis
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # Sort by frequency
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Huffman Encoding Procedure', fontsize=16, fontweight='bold')

    # Step 1: Original text
    axes[0, 0].text(0.1, 0.8, 'Step 1: Original Text', fontsize=14, fontweight='bold')
    axes[0, 0].text(0.1, 0.6, f'Text: {text}', fontsize=12, family='monospace')
    axes[0, 0].text(0.1, 0.4, f'Length: {len(text)} characters', fontsize=12)
    axes[0, 0].text(0.1, 0.2, f'ASCII bits: {len(text) * 8} bits', fontsize=12)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')

    # Step 2: Frequency count
    axes[0, 1].text(0.1, 0.9, 'Step 2: Count Frequencies', fontsize=14, fontweight='bold')
    y_pos = 0.75
    for char, count in sorted_freq:
        axes[0, 1].text(0.1, y_pos, f'{char}: {count}', fontsize=12, family='monospace')
        y_pos -= 0.1
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')

    # Step 3: Build tree (simplified visualization)
    axes[0, 2].text(0.1, 0.9, 'Step 3: Build Huffman Tree', fontsize=14, fontweight='bold')
    # Simple tree representation
    tree_text = """
           (11)
          /    \\
        A(5)   (6)
              /   \\
            B(2)  (4)
                 /   \\
               R(2)  (2)
                    /   \\
                  C(1) D(1)
    """
    axes[0, 2].text(0.05, 0.1, tree_text, fontsize=10, family='monospace')
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')

    # Step 4: Assign codes
    huffman_codes = {'A': '0', 'B': '10', 'R': '110', 'C': '1110', 'D': '1111'}
    axes[1, 0].text(0.1, 0.9, 'Step 4: Assign Codes', fontsize=14, fontweight='bold')
    y_pos = 0.75
    for char, code in huffman_codes.items():
        axes[1, 0].text(0.1, y_pos, f'{char}: {code}', fontsize=12, family='monospace')
        y_pos -= 0.1
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')

    # Step 5: Encode
    encoded = ''.join(huffman_codes[char] for char in text)
    axes[1, 1].text(0.1, 0.9, 'Step 5: Encode Text', fontsize=14, fontweight='bold')

    # Show character by character encoding
    y_pos = 0.75
    for char in text:
        axes[1, 1].text(0.1, y_pos, f'{char} → {huffman_codes[char]}', fontsize=10, family='monospace')
        y_pos -= 0.05

    axes[1, 1].text(0.1, 0.2, f'Encoded: {encoded}', fontsize=10, family='monospace')
    axes[1, 1].text(0.1, 0.1, f'Bits: {len(encoded)}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    # Step 6: Results
    original_bits = len(text) * 8
    compressed_bits = len(encoded)
    compression_ratio = (1 - compressed_bits/original_bits) * 100

    axes[1, 2].text(0.1, 0.9, 'Step 6: Results', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, f'Original: {original_bits} bits', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Compressed: {compressed_bits} bits', fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'Savings: {original_bits - compressed_bits} bits', fontsize=12)
    axes[1, 2].text(0.1, 0.3, f'Compression Ratio:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.2, f'{compression_ratio:.1f}%', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{images_dir}/huffman_procedure_steps.png", dpi=150, bbox_inches='tight')
    plt.close()

def generate_jpeg_procedure():
    """Step-by-step JPEG compression procedure"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('JPEG Compression Procedure', fontsize=16, fontweight='bold')

    # Create sample 8x8 block
    sample_block = np.array([
        [154, 123, 123, 123, 123, 123, 123, 136],
        [192, 180, 136, 154, 154, 154, 154, 136],
        [254, 198, 154, 154, 180, 154, 136, 123],
        [239, 180, 136, 180, 180, 166, 123, 123],
        [180, 154, 136, 167, 166, 149, 136, 136],
        [128, 136, 123, 136, 154, 180, 198, 154],
        [123, 105, 136, 154, 167, 154, 136, 149],
        [136, 154, 136, 149, 136, 136, 123, 136]
    ])

    # Step 1: Original block
    im1 = axes[0, 0].imshow(sample_block, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Step 1: Original 8×8 Block', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.6)

    # Step 2: Subtract 128 (DC shift)
    shifted_block = sample_block - 128
    im2 = axes[0, 1].imshow(shifted_block, cmap='RdBu', vmin=-128, vmax=127)
    axes[0, 1].set_title('Step 2: Subtract 128\n(Center around 0)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.6)

    # Step 3: DCT
    # Simplified DCT representation
    dct_block = np.array([
        [1024, -123, -36, -8, 0, 1, 0, 0],
        [-54, -92, -8, -2, 1, 0, 0, 0],
        [-42, -17, -6, -2, 0, 0, 0, 0],
        [-12, -9, -3, 0, 0, 0, 0, 0],
        [-7, -2, -1, 0, 0, 0, 0, 0],
        [-3, -1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    im3 = axes[0, 2].imshow(np.abs(dct_block), cmap='hot')
    axes[0, 2].set_title('Step 3: DCT Transform\n(Frequency Domain)', fontsize=12)
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.6)

    # Step 4: Quantization matrix
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

    im4 = axes[0, 3].imshow(quant_matrix, cmap='viridis')
    axes[0, 3].set_title('Step 4: Quantization Matrix\n(JPEG Standard)', fontsize=12)
    plt.colorbar(im4, ax=axes[0, 3], shrink=0.6)

    # Step 5: Quantized DCT
    quantized_dct = np.round(dct_block / quant_matrix)
    im5 = axes[1, 0].imshow(np.abs(quantized_dct), cmap='hot')
    axes[1, 0].set_title('Step 5: Quantized DCT\n(Lossy Compression)', fontsize=12)
    plt.colorbar(im5, ax=axes[1, 0], shrink=0.6)

    # Step 6: Zigzag ordering
    axes[1, 1].text(0.1, 0.9, 'Step 6: Zigzag Scan', fontsize=12, fontweight='bold')
    zigzag_text = """
    Start → 64, -8, -1, -9, ...

    Pattern:
    1→2  5→6  ...
    ↓  ↗ ↓  ↗
    3  4  7  8  ...
    ↓     ↓
    9→10 ...
    """
    axes[1, 1].text(0.05, 0.1, zigzag_text, fontsize=10, family='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    # Step 7: Entropy encoding
    axes[1, 2].text(0.1, 0.9, 'Step 7: Entropy Coding', fontsize=12, fontweight='bold')
    entropy_text = """
    DC Coefficient: 64
    → DPCM encoding

    AC Coefficients:
    → Run-length encoding
    → Huffman coding

    Final bitstream:
    11010001101...
    """
    axes[1, 2].text(0.05, 0.1, entropy_text, fontsize=10, family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    # Step 8: Reconstruction
    reconstructed = quantized_dct * quant_matrix + 128
    im8 = axes[1, 3].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[1, 3].set_title('Step 8: Reconstructed Block\n(After Decompression)', fontsize=12)
    plt.colorbar(im8, ax=axes[1, 3], shrink=0.6)

    plt.tight_layout()
    plt.savefig(f"{images_dir}/jpeg_procedure_steps.png", dpi=150, bbox_inches='tight')
    plt.close()

def generate_png_procedure():
    """Step-by-step PNG compression procedure"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PNG Compression Procedure (Lossless)', fontsize=16, fontweight='bold')

    # Sample scanline data
    scanline = [120, 122, 121, 123, 125, 124, 126, 128]

    # Step 1: Original data
    axes[0, 0].text(0.1, 0.9, 'Step 1: Original Scanline', fontsize=14, fontweight='bold')
    axes[0, 0].text(0.1, 0.7, 'Pixel values:', fontsize=12)
    pixel_str = ', '.join(map(str, scanline))
    axes[0, 0].text(0.1, 0.6, pixel_str, fontsize=12, family='monospace')
    axes[0, 0].text(0.1, 0.4, f'Size: {len(scanline)} bytes', fontsize=12)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')

    # Step 2: Filtering (Delta filter)
    filtered = [scanline[0]] + [scanline[i] - scanline[i-1] for i in range(1, len(scanline))]
    axes[0, 1].text(0.1, 0.9, 'Step 2: Delta Filtering', fontsize=14, fontweight='bold')
    axes[0, 1].text(0.1, 0.7, 'Formula: pixel[i] - pixel[i-1]', fontsize=12)
    axes[0, 1].text(0.1, 0.6, 'Original:', fontsize=12)
    axes[0, 1].text(0.1, 0.55, pixel_str, fontsize=10, family='monospace')
    axes[0, 1].text(0.1, 0.4, 'Filtered:', fontsize=12)
    filtered_str = ', '.join(map(str, filtered))
    axes[0, 1].text(0.1, 0.35, filtered_str, fontsize=10, family='monospace')
    axes[0, 1].text(0.1, 0.2, 'Smaller values = better compression!', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')

    # Step 3: DEFLATE compression
    axes[0, 2].text(0.1, 0.9, 'Step 3: DEFLATE Compression', fontsize=14, fontweight='bold')
    deflate_text = """
    1. LZ77 Algorithm:
       - Find repeated patterns
       - Replace with (distance, length)

    2. Huffman Coding:
       - Frequent symbols → short codes
       - Rare symbols → long codes

    Result: Compressed bitstream
    """
    axes[0, 2].text(0.05, 0.1, deflate_text, fontsize=11)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')

    # Step 4: Chunked format
    axes[1, 0].text(0.1, 0.9, 'Step 4: PNG File Structure', fontsize=14, fontweight='bold')
    chunk_text = """
    PNG Signature: 89 50 4E 47...

    IHDR Chunk: Image header
    - Width, height, bit depth
    - Color type, compression

    IDAT Chunk: Image data
    - Compressed pixel data

    IEND Chunk: End marker
    """
    axes[1, 0].text(0.05, 0.1, chunk_text, fontsize=10, family='monospace')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')

    # Step 5: Advantages
    axes[1, 1].text(0.1, 0.9, 'Step 5: PNG Advantages', fontsize=14, fontweight='bold')
    advantages_text = """
    ✓ Lossless compression
    ✓ Transparency support
    ✓ Good for graphics
    ✓ Cross-platform
    ✓ Patent-free

    Best for:
    • Screenshots
    • Line art
    • Images with text
    • Graphics with few colors
    """
    axes[1, 1].text(0.05, 0.1, advantages_text, fontsize=11)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    # Step 6: Comparison
    axes[1, 2].text(0.1, 0.9, 'Step 6: Typical Compression', fontsize=14, fontweight='bold')
    comp_text = """
    Image Type        Compression
    ─────────────────────────────
    Screenshots       60-80%
    Line art          70-90%
    Photographs       20-40%
    Graphics          50-70%

    Note: Actual ratios depend on
    image content and complexity
    """
    axes[1, 2].text(0.05, 0.2, comp_text, fontsize=10, family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{images_dir}/png_procedure_steps.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Generate all procedure demonstrations"""
    print("Generating compression procedure demonstrations...")

    print("1. Huffman encoding procedure...")
    generate_huffman_procedure()

    print("2. JPEG compression procedure...")
    generate_jpeg_procedure()

    print("3. PNG compression procedure...")
    generate_png_procedure()

    print("Procedure demonstrations generated successfully!")

if __name__ == "__main__":
    main()