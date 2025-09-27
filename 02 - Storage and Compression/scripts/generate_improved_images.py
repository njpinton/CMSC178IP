#!/usr/bin/env python3
"""
Generate improved visual demonstration images - focusing on graphics over text
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image, ImageDraw
import os

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

def create_improved_huffman_demo():
    """Create a highly visual Huffman encoding demonstration"""

    plt.figure(figsize=(16, 12))

    # Character frequency data
    text = "ABRACADABRA"
    freq = {'A': 5, 'B': 2, 'R': 2, 'C': 1, 'D': 1}
    huffman_codes = {'A': '0', 'B': '10', 'R': '110', 'C': '1110', 'D': '1111'}

    # Color scheme
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'R': '#45B7D1', 'C': '#96CEB4', 'D': '#FECA57'}

    # 1. Character frequency visualization with visual blocks
    plt.subplot(2, 3, 1)
    chars = list(freq.keys())
    frequencies = list(freq.values())

    bars = plt.bar(chars, frequencies, color=[colors[char] for char in chars],
                   edgecolor='black', linewidth=2, alpha=0.8)
    plt.title('Character Frequencies', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Characters', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Add visual frequency indicators
    for bar, freq_val in zip(bars, frequencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(freq_val), ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 2. Binary tree visualization
    plt.subplot(2, 3, 2)

    # Tree node positions (more spread out for clarity)
    nodes = {
        'root': (0.5, 0.9),
        'left': (0.25, 0.7),    # A
        'right': (0.75, 0.7),
        'B': (0.6, 0.5),
        'right2': (0.9, 0.5),
        'R': (0.8, 0.3),
        'right3': (1.0, 0.3),
        'C': (0.95, 0.1),
        'D': (1.05, 0.1)
    }

    # Draw tree connections
    connections = [
        ('root', 'left'), ('root', 'right'),
        ('right', 'B'), ('right', 'right2'),
        ('right2', 'R'), ('right2', 'right3'),
        ('right3', 'C'), ('right3', 'D')
    ]

    for start, end in connections:
        if start in nodes and end in nodes:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.7)

    # Draw nodes with character colors
    leaf_nodes = {'left': 'A', 'B': 'B', 'R': 'R', 'C': 'C', 'D': 'D'}

    for node, pos in nodes.items():
        x, y = pos
        if node in leaf_nodes:
            char = leaf_nodes[node]
            circle = plt.Circle((x, y), 0.06, color=colors[char],
                              edgecolor='black', linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(x, y, char, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        elif node != 'root':
            circle = plt.Circle((x, y), 0.04, color='lightgray',
                              edgecolor='black', linewidth=1)
            plt.gca().add_patch(circle)

    # Root node
    root_circle = plt.Circle(nodes['root'], 0.05, color='darkgray',
                           edgecolor='black', linewidth=2)
    plt.gca().add_patch(root_circle)

    plt.xlim(0, 1.2)
    plt.ylim(0, 1)
    plt.title('Huffman Tree', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    # 3. Bit length comparison
    plt.subplot(2, 3, 3)

    original_bits = len(text) * 8
    compressed_bits = sum(len(huffman_codes[char]) * freq[char] for char in freq)

    # Create visual bit representation
    methods = ['Original\\n(8 bits/char)', 'Huffman\\n(Variable)']
    bits = [original_bits, compressed_bits]
    bar_colors = ['#FF6B6B', '#4ECDC4']

    bars = plt.bar(methods, bits, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=2, width=0.6)

    plt.title('Compression Efficiency', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Total Bits Required', fontsize=14)

    for bar, bit_count in zip(bars, bits):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{bit_count}', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Add compression ratio
    compression_ratio = (1 - compressed_bits/original_bits) * 100
    plt.text(0.5, max(bits) * 0.8, f'{compression_ratio:.1f}% Savings',
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            transform=plt.gca().transData)

    # 4. Code length visualization
    plt.subplot(2, 3, 4)

    chars_ordered = sorted(huffman_codes.keys(), key=lambda x: len(huffman_codes[x]))
    code_lengths = [len(huffman_codes[char]) for char in chars_ordered]
    char_colors = [colors[char] for char in chars_ordered]

    bars = plt.bar(chars_ordered, code_lengths, color=char_colors,
                  alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Code Lengths', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Characters', fontsize=14)
    plt.ylabel('Bits per Character', fontsize=14)

    for bar, length in zip(bars, code_lengths):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                str(length), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # 5. Visual encoding example
    plt.subplot(2, 3, 5)

    # Show encoding of "ABRA"
    example_text = "ABRA"
    encoded_bits = "".join(huffman_codes[char] for char in example_text)

    # Visual representation of bits
    bit_positions = np.arange(len(encoded_bits))
    bit_values = [int(bit) for bit in encoded_bits]
    bit_colors = ['white' if bit == 0 else 'black' for bit in bit_values]

    bars = plt.bar(bit_positions, [1]*len(encoded_bits),
                  color=bit_colors, edgecolor='gray', linewidth=1, width=0.8)

    plt.title(f'"{example_text}" → {encoded_bits}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Bit Value', fontsize=14)
    plt.xlabel('Bit Position', fontsize=14)
    plt.yticks([0, 1])

    # Add bit labels
    for i, bit in enumerate(encoded_bits):
        plt.text(i, 0.5, bit, ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='black' if bit == '0' else 'white')

    # 6. Frequency vs Code Length relationship
    plt.subplot(2, 3, 6)

    freq_values = [freq[char] for char in chars]
    code_lens = [len(huffman_codes[char]) for char in chars]
    char_colors_scatter = [colors[char] for char in chars]

    scatter = plt.scatter(freq_values, code_lens,
                         c=char_colors_scatter, s=200, alpha=0.8,
                         edgecolors='black', linewidth=2)

    # Add character labels
    for char in chars:
        plt.annotate(char, (freq[char], len(huffman_codes[char])),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold')

    plt.title('Frequency vs Code Length', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Character Frequency', fontsize=14)
    plt.ylabel('Code Length (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)

    save_figure('huffman_encoding_demo.png', figsize=(16, 12))

def create_improved_rle_demo():
    """Create a highly visual RLE demonstration"""

    plt.figure(figsize=(14, 10))

    # Sample data with clear runs
    original_data = "AAABBBBCCDDDDDDAAA"

    # Color mapping for characters
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#96CEB4'}

    # 1. Original data visualization
    plt.subplot(3, 2, (1, 2))

    positions = np.arange(len(original_data))
    char_colors = [colors[char] for char in original_data]

    bars = plt.bar(positions, [1]*len(original_data),
                  color=char_colors, edgecolor='black', linewidth=1, width=0.9)

    plt.title(f'Original Data: "{original_data}"', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Character', fontsize=14)

    # Add character labels
    for i, char in enumerate(original_data):
        plt.text(i, 0.5, char, ha='center', va='center',
                fontsize=10, fontweight='bold')

    plt.xticks(positions[::2])  # Show every other position
    plt.yticks([])

    # 2. Run identification
    plt.subplot(3, 2, 3)

    # Identify runs
    runs = []
    current_char = original_data[0]
    current_count = 1

    for char in original_data[1:]:
        if char == current_char:
            current_count += 1
        else:
            runs.append((current_count, current_char))
            current_char = char
            current_count = 1
    runs.append((current_count, current_char))

    # Visualize runs
    run_positions = np.arange(len(runs))
    run_counts = [count for count, _ in runs]
    run_chars = [char for _, char in runs]
    run_colors = [colors[char] for char in run_chars]

    bars = plt.bar(run_positions, run_counts, color=run_colors,
                  alpha=0.8, edgecolor='black', linewidth=2)

    plt.title('Identified Runs', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Run Number', fontsize=14)
    plt.ylabel('Run Length', fontsize=14)

    # Add run labels
    for i, (count, char) in enumerate(runs):
        plt.text(i, count + 0.2, f'{count}{char}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 3. RLE encoded representation
    plt.subplot(3, 2, 4)

    rle_string = "".join(f"{count}{char}" for count, char in runs)

    # Create visual blocks for RLE
    rle_positions = np.arange(len(rle_string))

    # Alternate colors for count and character
    rle_colors = []
    for i, c in enumerate(rle_string):
        if c.isdigit():
            rle_colors.append('lightblue')
        else:
            rle_colors.append(colors[c])

    bars = plt.bar(rle_positions, [1]*len(rle_string),
                  color=rle_colors, edgecolor='black', linewidth=1)

    plt.title(f'RLE Encoded: "{rle_string}"', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Position in Encoded String', fontsize=14)

    # Add character labels
    for i, c in enumerate(rle_string):
        plt.text(i, 0.5, c, ha='center', va='center',
                fontsize=12, fontweight='bold')

    plt.yticks([])

    # 4. Compression comparison
    plt.subplot(3, 2, 5)

    original_length = len(original_data)
    compressed_length = len(rle_string)
    compression_ratio = (1 - compressed_length/original_length) * 100

    methods = ['Original', 'RLE Compressed']
    lengths = [original_length, compressed_length]
    bar_colors = ['#FF6B6B', '#4ECDC4']

    bars = plt.bar(methods, lengths, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=2)

    plt.title('Compression Results', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('String Length', fontsize=14)

    for bar, length in zip(bars, lengths):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(length), ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Add compression ratio
    plt.text(0.5, max(lengths) * 0.7, f'{compression_ratio:.1f}% Savings',
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

    # 5. Step-by-step process
    plt.subplot(3, 2, 6)

    # Show the RLE algorithm steps visually
    steps = [
        "1. Scan left to right",
        "2. Count consecutive chars",
        "3. Write count + character",
        "4. Repeat until end"
    ]

    step_colors = ['#FFE5CC', '#CCE5FF', '#E5CCFF', '#E5FFCC']

    for i, (step, color) in enumerate(zip(steps, step_colors)):
        rect = mpatches.Rectangle((0.1, 0.8-i*0.2), 0.8, 0.15,
                                 facecolor=color, edgecolor='black', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(0.5, 0.875-i*0.2, step, ha='center', va='center',
                fontsize=12, fontweight='bold')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('RLE Algorithm Steps', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    save_figure('rle_encoding_demo.png', figsize=(14, 10))

def create_improved_storage_formats():
    """Create improved storage format comparison with visual elements"""

    plt.figure(figsize=(16, 10))

    # Sample data for different formats
    formats = ['BMP', 'PNG', 'JPEG-High', 'JPEG-Medium', 'JPEG-Low']
    file_sizes = [2048, 856, 512, 256, 128]  # KB
    quality_scores = [100, 100, 95, 85, 70]  # Quality percentage

    format_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    # 1. File size comparison
    plt.subplot(2, 3, 1)

    bars = plt.bar(formats, file_sizes, color=format_colors, alpha=0.8,
                  edgecolor='black', linewidth=2)

    plt.title('File Size Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('File Size (KB)', fontsize=14)
    plt.xticks(rotation=45)

    for bar, size in zip(bars, file_sizes):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{size}KB', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 2. Quality vs Size scatter plot
    plt.subplot(2, 3, 2)

    scatter = plt.scatter(file_sizes, quality_scores, c=format_colors,
                         s=300, alpha=0.8, edgecolors='black', linewidth=2)

    # Add format labels
    for i, fmt in enumerate(formats):
        plt.annotate(fmt, (file_sizes[i], quality_scores[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')

    plt.title('Quality vs File Size', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('File Size (KB)', fontsize=14)
    plt.ylabel('Quality Score', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 3. Compression efficiency
    plt.subplot(2, 3, 3)

    # Calculate compression ratios relative to BMP
    bmp_size = file_sizes[0]
    compression_ratios = [bmp_size / size for size in file_sizes]

    bars = plt.bar(formats, compression_ratios, color=format_colors, alpha=0.8,
                  edgecolor='black', linewidth=2)

    plt.title('Compression Ratio vs BMP', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Compression Factor', fontsize=14)
    plt.xticks(rotation=45)

    for bar, ratio in zip(bars, compression_ratios):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{ratio:.1f}x', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 4. Format characteristics matrix
    plt.subplot(2, 3, (4, 6))

    # Characteristics matrix
    characteristics = {
        'Compression': [0, 3, 4, 4, 4],  # 0=None, 4=High
        'Quality': [5, 5, 4, 3, 2],      # 5=Perfect, 1=Poor
        'File Size': [5, 3, 2, 2, 1],   # 5=Large, 1=Small
        'Speed': [5, 3, 4, 4, 5]         # 5=Fast, 1=Slow
    }

    # Create heatmap-style visualization
    char_names = list(characteristics.keys())
    char_data = np.array(list(characteristics.values()))

    # Create custom visualization
    for i, format_name in enumerate(formats):
        y_offset = i * 1.2

        # Format name
        rect = mpatches.Rectangle((0, y_offset), 2, 1,
                                 facecolor=format_colors[i],
                                 edgecolor='black', linewidth=2, alpha=0.8)
        plt.gca().add_patch(rect)
        plt.text(1, y_offset + 0.5, format_name, ha='center', va='center',
                fontsize=14, fontweight='bold')

        # Characteristics bars
        for j, char in enumerate(char_names):
            x_pos = 3 + j * 2.5
            value = char_data[j, i]

            # Create mini bar chart for each characteristic
            bar_height = value / 5.0 * 0.8  # Scale to fit

            rect = mpatches.Rectangle((x_pos, y_offset + 0.1), 0.8, bar_height,
                                     facecolor='steelblue', alpha=0.7,
                                     edgecolor='black', linewidth=1)
            plt.gca().add_patch(rect)

            # Add value label
            plt.text(x_pos + 0.4, y_offset + bar_height + 0.1, str(value),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add characteristic headers
    for j, char in enumerate(char_names):
        x_pos = 3 + j * 2.5
        plt.text(x_pos + 0.4, len(formats) * 1.2 + 0.2, char,
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                rotation=0)

    plt.xlim(-0.5, 13)
    plt.ylim(-0.5, len(formats) * 1.2 + 1)
    plt.title('Format Characteristics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    save_figure('storage_formats_comparison.png', figsize=(16, 10))

if __name__ == "__main__":
    print("Generating improved visual demonstration images...")

    print("1. Improved Huffman encoding demo...")
    create_improved_huffman_demo()

    print("2. Improved RLE demo...")
    create_improved_rle_demo()

    print("3. Improved storage formats comparison...")
    create_improved_storage_formats()

    print("Improved visual images generated successfully!")