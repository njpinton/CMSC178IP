"""
Image Segmentation Visualizations
Demonstrates semantic and instance segmentation concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
output_dir = Path(__file__).parent.parent / 'figures'
output_dir.mkdir(exist_ok=True)


def create_segmentation_types():
    """Compare semantic vs instance segmentation"""
    np.random.seed(42)

    # Create simple image with multiple objects
    img = np.ones((200, 300, 3)) * 0.9

    # Add some objects (circles)
    y1, x1 = np.ogrid[-100:100, -150:150]

    # Object 1 - person (top left)
    mask1 = (x1 + 80)**2 + (y1 + 50)**2 <= 30**2
    img[mask1] = [0.8, 0.6, 0.5]

    # Object 2 - person (top right)
    mask2 = (x1 - 80)**2 + (y1 + 50)**2 <= 30**2
    img[mask2] = [0.75, 0.55, 0.45]

    # Object 3 - car (bottom)
    mask3 = (x1)**2 + (y1 - 40)**2 <= 40**2
    img[mask3] = [0.3, 0.5, 0.7]

    # Semantic segmentation
    semantic = np.zeros((200, 300, 3))
    semantic[mask1 | mask2] = [1.0, 0.5, 0.5]  # All persons same color
    semantic[mask3] = [0.5, 0.5, 1.0]  # Car different color
    semantic[~(mask1 | mask2 | mask3)] = [0.2, 0.2, 0.2]  # Background

    # Instance segmentation
    instance = np.zeros((200, 300, 3))
    instance[mask1] = [1.0, 0.3, 0.3]  # Person 1
    instance[mask2] = [0.3, 1.0, 0.3]  # Person 2
    instance[mask3] = [0.3, 0.3, 1.0]  # Car
    instance[~(mask1 | mask2 | mask3)] = [0.2, 0.2, 0.2]  # Background

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(semantic)
    axes[1].set_title('Semantic Segmentation\n(Person vs Car)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(instance)
    axes[2].set_title('Instance Segmentation\n(Each Object Unique)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_types.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_unet_architecture():
    """Create U-Net architecture diagram based on reference image"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define layers with (x, y_bottom, width, height, channels, size_label)
    # Encoder path - going down and to the right
    encoder_specs = [
        (0.05, 0.35, 0.04, 0.30, '64', '572×572'),
        (0.12, 0.40, 0.03, 0.20, '128', '280×280'),
        (0.18, 0.43, 0.025, 0.14, '256', '136×136'),
        (0.23, 0.45, 0.02, 0.10, '512', '64×64'),
    ]

    # Bottleneck
    bottleneck_spec = (0.27, 0.46, 0.015, 0.08, '1024', '32×32')

    # Decoder path - going up and to the right
    decoder_specs = [
        (0.31, 0.45, 0.02, 0.10, '512', '56×56'),
        (0.36, 0.43, 0.025, 0.14, '256', '104×104'),
        (0.42, 0.40, 0.03, 0.20, '128', '200×200'),
        (0.49, 0.35, 0.04, 0.30, '64', '388×388'),
    ]

    # Color scheme
    encoder_color = '#4285F4'  # Blue
    decoder_color = '#34A853'  # Green
    bottleneck_color = '#9333EA'  # Purple

    # Draw encoder blocks
    encoder_blocks = []
    for x, y, w, h, channels, size in encoder_specs:
        # Draw two blocks stacked (conv + conv)
        rect1 = plt.Rectangle((x, y), w, h, facecolor=encoder_color,
                              edgecolor='black', linewidth=1.5)
        rect2 = plt.Rectangle((x + w*1.1, y), w, h, facecolor=encoder_color,
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        # Add channel label on top
        ax.text(x + w, y + h + 0.02, channels, ha='center', va='bottom',
               fontsize=9, fontweight='bold')
        # Add size label below
        ax.text(x + w, y - 0.02, size, ha='center', va='top',
               fontsize=7)

        encoder_blocks.append((x + w*1.1, y + h/2))  # Store center for connections

    # Draw bottleneck
    x, y, w, h, channels, size = bottleneck_spec
    rect1 = plt.Rectangle((x, y), w, h, facecolor=bottleneck_color,
                          edgecolor='black', linewidth=1.5)
    rect2 = plt.Rectangle((x + w*1.1, y), w, h, facecolor=bottleneck_color,
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.text(x + w, y + h + 0.02, channels, ha='center', va='bottom',
           fontsize=9, fontweight='bold')
    ax.text(x + w, y - 0.02, size, ha='center', va='top', fontsize=7)
    bottleneck_center = (x + w*1.1, y + h/2)

    # Draw decoder blocks
    decoder_blocks = []
    for x, y, w, h, channels, size in decoder_specs:
        rect1 = plt.Rectangle((x, y), w, h, facecolor=decoder_color,
                              edgecolor='black', linewidth=1.5)
        rect2 = plt.Rectangle((x + w*1.1, y), w, h, facecolor=decoder_color,
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.text(x + w, y + h + 0.02, channels, ha='center', va='bottom',
               fontsize=9, fontweight='bold')
        ax.text(x + w, y - 0.02, size, ha='center', va='top', fontsize=7)

        decoder_blocks.append((x, y + h/2))

    # Draw max pooling arrows (red) - encoder downsampling
    for i in range(len(encoder_blocks)):
        if i < len(encoder_blocks) - 1:
            x1, y1 = encoder_blocks[i]
            x2, y2 = encoder_specs[i+1][0], encoder_specs[i+1][1] + encoder_specs[i+1][3]/2
            ax.arrow(x1 + 0.01, y1, x2 - x1 - 0.02, y2 - y1,
                    head_width=0.015, head_length=0.008, fc='red', ec='red', linewidth=2)
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.02, 'max pool\n2×2', ha='center',
                   fontsize=6, color='red', fontweight='bold')

    # Last encoder to bottleneck
    x1, y1 = encoder_blocks[-1]
    x2, y2 = bottleneck_spec[0], bottleneck_spec[1] + bottleneck_spec[3]/2
    ax.arrow(x1 + 0.01, y1, x2 - x1 - 0.02, y2 - y1,
            head_width=0.015, head_length=0.008, fc='red', ec='red', linewidth=2)
    ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.02, 'max pool\n2×2', ha='center',
           fontsize=6, color='red', fontweight='bold')

    # Draw up-convolution arrows (green) - decoder upsampling
    x1, y1 = bottleneck_center
    x2, y2 = decoder_blocks[0]
    ax.arrow(x1 + 0.01, y1, x2 - x1 - 0.02, y2 - y1,
            head_width=0.015, head_length=0.008, fc='green', ec='green', linewidth=2)
    ax.text((x1 + x2)/2, (y1 + y2)/2 - 0.02, 'up-conv\n2×2', ha='center',
           fontsize=6, color='green', fontweight='bold')

    for i in range(len(decoder_blocks) - 1):
        x1, y1 = decoder_specs[i][0] + decoder_specs[i][2]*2.2, decoder_specs[i][1] + decoder_specs[i][3]/2
        x2, y2 = decoder_blocks[i+1]
        ax.arrow(x1 + 0.01, y1, x2 - x1 - 0.02, y2 - y1,
                head_width=0.015, head_length=0.008, fc='green', ec='green', linewidth=2)
        ax.text((x1 + x2)/2, (y1 + y2)/2 - 0.02, 'up-conv\n2×2', ha='center',
               fontsize=6, color='green', fontweight='bold')

    # Draw skip connections (copy and crop) - gray dashed
    for i in range(len(encoder_blocks)):
        x1, y1 = encoder_blocks[i]
        x2, y2 = decoder_blocks[-(i+1)]

        # Curved connection
        mid_x = (x1 + x2) / 2
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray',
                                 linestyle='--', connectionstyle='arc3,rad=0.3'))
        ax.text(mid_x, max(y1, y2) + 0.03, 'copy and crop', ha='center',
               fontsize=6, color='gray', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw conv 3×3 ReLU labels (purple arrows)
    for i, (x, y, w, h, _, _) in enumerate(encoder_specs):
        ax.arrow(x - 0.01, y + h/2, 0.03, 0, head_width=0.01, head_length=0.005,
                fc='purple', ec='purple', linewidth=1.5)
        if i == 0:
            ax.text(x - 0.015, y + h/2 + 0.03, 'conv 3×3, ReLU', ha='right', fontsize=7,
                   color='purple', fontweight='bold')

    # Add input/output labels
    ax.text(0.03, 0.50, 'Input\nImage\ntile', ha='center', va='center', fontsize=9,
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    last_x = decoder_specs[-1][0] + decoder_specs[-1][2]*2.2
    ax.text(last_x + 0.04, 0.50, 'Output\nSegment\nMap', ha='center', va='center', fontsize=9,
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # Add final 1×1 conv
    ax.arrow(last_x + 0.01, 0.50, 0.02, 0, head_width=0.01, head_length=0.005,
            fc='blue', ec='blue', linewidth=2)
    ax.text(last_x + 0.012, 0.52, 'conv 1×1', ha='center', fontsize=7,
           color='blue', fontweight='bold')

    # Legend
    legend_y = 0.08
    ax.plot([0.1, 0.13], [legend_y, legend_y], 'purple', linewidth=2, label='Conv 3×3, ReLU')
    ax.plot([0.2, 0.23], [legend_y, legend_y], 'gray', linewidth=2, linestyle='--', label='Copy and crop')
    ax.plot([0.32, 0.35], [legend_y, legend_y], 'green', linewidth=2, label='Up-conv 2×2')
    ax.plot([0.44, 0.47], [legend_y, legend_y], 'red', linewidth=2, label='Max pool 2×2')
    ax.plot([0.55, 0.58], [legend_y, legend_y], 'blue', linewidth=2, label='Conv 1×1')

    ax.legend(loc='lower center', ncol=5, fontsize=8, framealpha=0.9)

    ax.set_xlim(0, 0.62)
    ax.set_ylim(0.05, 0.75)
    ax.axis('off')
    ax.set_title('U-Net Architecture', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'unet_architecture.png', dpi=200, bbox_inches='tight')
    plt.close()


def create_segmentation_example():
    """Create segmentation prediction example"""
    np.random.seed(42)

    # Create synthetic image
    img = np.ones((256, 256, 3)) * 0.8

    # Add some structured regions
    y, x = np.ogrid[-128:128, -128:128]

    # Region 1 - circle
    mask1 = (x + 60)**2 + (y + 60)**2 <= 40**2
    img[mask1] = [0.9, 0.5, 0.5]

    # Region 2 - square
    mask2 = (abs(x - 60) <= 40) & (abs(y + 60) <= 40)
    img[mask2] = [0.5, 0.9, 0.5]

    # Region 3 - triangle-like
    mask3 = (abs(x) <= 40) & (y >= 20) & (y <= 100)
    img[mask3] = [0.5, 0.5, 0.9]

    # Ground truth segmentation
    gt = np.zeros((256, 256))
    gt[mask1] = 1
    gt[mask2] = 2
    gt[mask3] = 3

    # Predicted segmentation (with some errors)
    pred = gt.copy()
    # Add some noise to predictions
    noise_mask = np.random.rand(256, 256) < 0.05
    pred[noise_mask] = np.random.randint(0, 4, noise_mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(gt, cmap='tab10', vmin=0, vmax=9)
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    im2 = axes[2].imshow(pred, cmap='tab10', vmin=0, vmax=9)
    axes[2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_example.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_iou_visualization():
    """Create IoU (Intersection over Union) visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    scenarios = [
        ('High IoU (0.85)', 0.85),
        ('Medium IoU (0.50)', 0.50),
        ('Low IoU (0.25)', 0.25),
    ]

    for ax, (title, iou) in zip(axes, scenarios):
        # Create ground truth box
        gt_box = plt.Rectangle((0.2, 0.2), 0.4, 0.4,
                              fill=False, edgecolor='green', linewidth=3,
                              linestyle='--', label='Ground Truth')

        # Create prediction box based on IoU
        if iou > 0.7:
            pred_box = plt.Rectangle((0.25, 0.25), 0.4, 0.4,
                                    fill=False, edgecolor='blue', linewidth=3,
                                    label='Prediction')
        elif iou > 0.4:
            pred_box = plt.Rectangle((0.3, 0.3), 0.4, 0.4,
                                    fill=False, edgecolor='blue', linewidth=3,
                                    label='Prediction')
        else:
            pred_box = plt.Rectangle((0.4, 0.4), 0.3, 0.3,
                                    fill=False, edgecolor='blue', linewidth=3,
                                    label='Prediction')

        ax.add_patch(gt_box)
        ax.add_patch(pred_box)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add IoU text
        ax.text(0.5, 0.05, f'IoU = {iou:.2f}',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.suptitle('Intersection over Union (IoU) Metric', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Generating segmentation figures...")
    create_segmentation_types()
    print("  ✓ Segmentation types comparison")
    create_unet_architecture()
    print("  ✓ U-Net architecture")
    create_segmentation_example()
    print("  ✓ Segmentation example")
    create_iou_visualization()
    print("  ✓ IoU visualization")
    print("Done!")
