"""
Advanced techniques for Computer Vision and Deep Learning I.
Generates visualizations for CNNs, convolution operations, and pooling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import convolve2d
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Get figures directory
figures_dir = Path(__file__).parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)

def generate_convolution_operation():
    """Visualize convolution operation step by step."""
    # Create a simple input image
    input_image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0],
        [2, 0, 1, 2, 3],
        [1, 2, 0, 1, 2],
        [3, 1, 2, 0, 1]
    ])

    # Edge detection kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    # Perform convolution
    output = convolve2d(input_image, kernel, mode='valid')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    im1 = axes[0].imshow(input_image, cmap='gray', interpolation='nearest')
    axes[0].set_title('Input Image (5×5)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f'{input_image[i, j]}', ha='center', va='center',
                        color='red', fontsize=10, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Kernel
    im2 = axes[1].imshow(kernel, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=8)
    axes[1].set_title('Kernel/Filter (3×3)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel[i, j]}', ha='center', va='center',
                        color='black' if kernel[i, j] != 8 else 'white',
                        fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Output
    im3 = axes[2].imshow(output, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Output Feature Map (3×3)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(output.shape[1]))
    axes[2].set_yticks(range(output.shape[0]))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            axes[2].text(j, i, f'{output[i, j]:.0f}', ha='center', va='center',
                        color='white', fontsize=10, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(figures_dir / "06_convolution_operation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 06_convolution_operation.png")

def generate_pooling_operations():
    """Visualize max pooling and average pooling."""
    # Create feature map
    feature_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 3],
        [2, 1, 4, 7],
        [3, 2, 6, 5]
    ])

    # Max pooling
    max_pool = np.array([
        [np.max(feature_map[0:2, 0:2]), np.max(feature_map[0:2, 2:4])],
        [np.max(feature_map[2:4, 0:2]), np.max(feature_map[2:4, 2:4])]
    ])

    # Average pooling
    avg_pool = np.array([
        [np.mean(feature_map[0:2, 0:2]), np.mean(feature_map[0:2, 2:4])],
        [np.mean(feature_map[2:4, 0:2]), np.mean(feature_map[2:4, 2:4])]
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input feature map
    im1 = axes[0].imshow(feature_map, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Input Feature Map (4×4)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{feature_map[i, j]}', ha='center', va='center',
                        fontsize=11, fontweight='bold')

    # Draw pooling regions
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            rect = Rectangle((j-0.5, i-0.5), 2, 2, linewidth=2,
                           edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Max pooling
    im2 = axes[1].imshow(max_pool, cmap='Reds', interpolation='nearest')
    axes[1].set_title('Max Pooling (2×2)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(2))
    axes[1].set_yticks(range(2))
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{max_pool[i, j]:.0f}', ha='center', va='center',
                        fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Average pooling
    im3 = axes[2].imshow(avg_pool, cmap='Greens', interpolation='nearest')
    axes[2].set_title('Average Pooling (2×2)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(2))
    axes[2].set_yticks(range(2))
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f'{avg_pool[i, j]:.1f}', ha='center', va='center',
                        fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(figures_dir / "07_pooling_operations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 07_pooling_operations.png")

def generate_cnn_architecture():
    """Generate CNN architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Layer specifications
    layers = [
        {'name': 'Input\n32×32×3', 'x': 1, 'size': (1.2, 1.2), 'color': '#2E86AB'},
        {'name': 'Conv1\n28×28×32', 'x': 2.8, 'size': (1.0, 1.0), 'color': '#A23B72'},
        {'name': 'Pool1\n14×14×32', 'x': 4.3, 'size': (0.8, 0.8), 'color': '#F18F01'},
        {'name': 'Conv2\n10×10×64', 'x': 5.8, 'size': (0.9, 0.9), 'color': '#A23B72'},
        {'name': 'Pool2\n5×5×64', 'x': 7.3, 'size': (0.7, 0.7), 'color': '#F18F01'},
        {'name': 'Flatten\n1600', 'x': 8.8, 'size': (0.3, 1.5), 'color': '#C73E1D'},
        {'name': 'FC1\n128', 'x': 10.0, 'size': (0.3, 1.2), 'color': '#6A4C93'},
        {'name': 'FC2\n10', 'x': 11.2, 'size': (0.3, 0.8), 'color': '#1982C4'}
    ]

    # Draw layers
    for layer in layers:
        width, height = layer['size']
        x, y = layer['x'], 3
        rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.05", linewidth=2,
                             edgecolor='black', facecolor=layer['color'], alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, layer['name'], ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    # Draw arrows
    for i in range(len(layers) - 1):
        start_x = layers[i]['x'] + layers[i]['size'][0]/2
        end_x = layers[i+1]['x'] - layers[i+1]['size'][0]/2
        ax.annotate('', xy=(end_x, 3), xytext=(start_x, 3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Output
    ax.text(12.5, 3, 'Output\n(Classes)', ha='center', va='center',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.annotate('', xy=(12.2, 3), xytext=(11.5, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Title
    ax.text(7, 5.5, 'Convolutional Neural Network Architecture',
           fontsize=16, ha='center', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#A23B72', label='Convolution'),
        mpatches.Patch(color='#F18F01', label='Pooling'),
        mpatches.Patch(color='#6A4C93', label='Fully Connected')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    plt.savefig(figures_dir / "08_cnn_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 08_cnn_architecture.png")

def generate_feature_maps_visualization():
    """Visualize multiple feature maps from convolution layers."""
    np.random.seed(42)

    # Simulate feature maps
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for idx, ax in enumerate(axes.flat):
        # Create synthetic feature map
        feature_map = np.random.randn(16, 16)
        feature_map = np.abs(feature_map) * np.exp(-((np.arange(16)[:, None] - 8)**2 +
                                                       (np.arange(16)[None, :] - 8)**2) / 50)

        im = ax.imshow(feature_map, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Feature Map {idx+1}', fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Learned Feature Maps from Convolutional Layer',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(figures_dir / "09_feature_maps.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 09_feature_maps.png")

def generate_learning_curves():
    """Generate training and validation learning curves."""
    epochs = np.arange(1, 51)

    # Simulate training curves
    train_loss = 2.5 * np.exp(-epochs / 10) + 0.1 + np.random.randn(50) * 0.05
    val_loss = 2.5 * np.exp(-epochs / 12) + 0.15 + np.random.randn(50) * 0.08

    train_acc = 1 - (0.9 * np.exp(-epochs / 8)) + np.random.randn(50) * 0.02
    val_acc = 1 - (0.9 * np.exp(-epochs / 10)) + np.random.randn(50) * 0.03

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, train_loss, linewidth=2.5, label='Training Loss', color='#2E86AB')
    axes[0].plot(epochs, val_loss, linewidth=2.5, label='Validation Loss', color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, linewidth=2.5, label='Training Accuracy', color='#2E86AB')
    axes[1].plot(epochs, val_acc, linewidth=2.5, label='Validation Accuracy', color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "10_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 10_learning_curves.png")

def main():
    """Generate all advanced technique figures."""
    print("Generating advanced technique figures...")
    print("-" * 40)

    generate_convolution_operation()
    generate_pooling_operations()
    generate_cnn_architecture()
    generate_feature_maps_visualization()
    generate_learning_curves()

    print("-" * 40)
    print("Advanced techniques figures complete!")

if __name__ == "__main__":
    main()
