"""
Object Detection Visualizations
Demonstrates YOLO and Faster R-CNN concepts
Using real images from scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from skimage import data, color, transform, draw
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
output_dir = Path(__file__).parent.parent / 'figures'
output_dir.mkdir(exist_ok=True)


def create_detection_example():
    """Create object detection visualization with real image and bounding boxes"""
    np.random.seed(42)

    # Use a real image from scikit-image
    img = data.coffee()  # Real photo with coffee cup

    # Define detected objects with bounding boxes [x, y, w, h, class, confidence, color]
    # Manually positioned to look like realistic detections
    objects = [
        (100, 50, 180, 200, 'cup', 0.94, '#FF6B6B'),
        (320, 80, 150, 140, 'plate', 0.88, '#4ECDC4'),
        (50, 240, 120, 80, 'spoon', 0.76, '#95E1D3'),
        (280, 220, 100, 90, 'saucer', 0.85, '#F38181'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Draw bounding boxes
    for x, y, w, h, cls, conf, color in objects:
        rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label with confidence score
        label = f'{cls}: {conf:.2f}'
        ax.text(x, y - 8, label, fontsize=11, fontweight='bold',
               color='white', bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

    ax.set_title('Object Detection Example (Multi-Object)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add detection info box
    info_text = f"Detected: {len(objects)} objects\nAvg Confidence: {np.mean([o[5] for o in objects]):.2f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'detection_example.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_yolo_grid():
    """Visualize YOLO grid-based detection"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid
    grid_size = 7
    img_size = 448
    cell_size = img_size / grid_size

    # Background
    img = np.ones((img_size, img_size, 3)) * 0.9
    ax.imshow(img, extent=[0, img_size, img_size, 0])

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(i * cell_size, color='gray', linewidth=1, alpha=0.5)
        ax.axvline(i * cell_size, color='gray', linewidth=1, alpha=0.5)

    # Add sample object (car)
    obj_x, obj_y = 150, 200
    obj_w, obj_h = 180, 120

    rect = patches.Rectangle((obj_x, obj_y), obj_w, obj_h,
                            linewidth=3, edgecolor='red',
                            facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Highlight responsible grid cells
    grid_x_start = int(obj_x / cell_size)
    grid_y_start = int(obj_y / cell_size)
    grid_x_end = int((obj_x + obj_w) / cell_size)
    grid_y_end = int((obj_y + obj_h) / cell_size)

    # Center cell
    center_x = int((obj_x + obj_w/2) / cell_size)
    center_y = int((obj_y + obj_h/2) / cell_size)

    highlight = patches.Rectangle((center_x * cell_size, center_y * cell_size),
                                cell_size, cell_size,
                                facecolor='yellow', alpha=0.3, edgecolor='orange',
                                linewidth=3)
    ax.add_patch(highlight)

    # Draw center point
    center_px = obj_x + obj_w/2
    center_py = obj_y + obj_h/2
    ax.plot(center_px, center_py, 'ro', markersize=10, label='Object Center')

    ax.text(obj_x + obj_w/2, obj_y - 15, 'Car (conf: 0.93)',
           fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.7, edgecolor='black'),
           color='white')

    ax.text(center_x * cell_size + cell_size/2, center_y * cell_size + cell_size/2,
           'Responsible\nCell', fontsize=9, ha='center', va='center',
           fontweight='bold')

    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.set_title(f'YOLO: {grid_size}×{grid_size} Grid Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_rcnn_pipeline():
    """Create Faster R-CNN pipeline diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))

    stages = [
        {'name': 'Input\nImage', 'x': 0.05, 'color': '#3498db'},
        {'name': 'CNN\nBackbone\n(ResNet)', 'x': 0.18, 'color': '#e74c3c'},
        {'name': 'Feature\nMaps', 'x': 0.31, 'color': '#f39c12'},
        {'name': 'Region\nProposal\nNetwork', 'x': 0.44, 'color': '#9b59b6'},
        {'name': 'RoI\nPooling', 'x': 0.57, 'color': '#1abc9c'},
        {'name': 'Fully\nConnected\nLayers', 'x': 0.70, 'color': '#e67e22'},
        {'name': 'Class +\nBBox\nOutput', 'x': 0.83, 'color': '#27ae60'},
    ]

    box_width = 0.10
    box_height = 0.25
    y_center = 0.5

    for stage in stages:
        x = stage['x']
        rect = plt.Rectangle((x, y_center - box_height/2), box_width, box_height,
                            facecolor=stage['color'], edgecolor='black',
                            linewidth=2, alpha=0.8)
        ax.add_patch(rect)

        ax.text(x + box_width/2, y_center, stage['name'],
               ha='center', va='center', fontsize=9, fontweight='bold',
               color='white')

    # Draw arrows
    for i in range(len(stages) - 1):
        x1 = stages[i]['x'] + box_width
        x2 = stages[i+1]['x']
        ax.arrow(x1, y_center, x2 - x1 - 0.01, 0,
                head_width=0.03, head_length=0.015, fc='black', ec='black',
                linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    ax.set_title('Faster R-CNN Detection Pipeline', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'faster_rcnn_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_nms_visualization():
    """Create Non-Maximum Suppression visualization"""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Create background
    img = np.ones((300, 400, 3)) * 0.85

    # Overlapping bounding boxes (before NMS)
    boxes_before = [
        (100, 80, 150, 180, 0.95, '#FF6B6B'),
        (110, 85, 150, 180, 0.89, '#FFB6B6'),
        (105, 75, 150, 180, 0.91, '#FFA6A6'),
        (115, 90, 150, 180, 0.87, '#FF9696'),
        (95, 82, 150, 180, 0.93, '#FF8686'),
    ]

    # After NMS (only best box)
    boxes_after = [
        (100, 80, 150, 180, 0.95, '#FF6B6B'),
    ]

    # Plot before NMS
    ax1.imshow(img)
    for x, y, w, h, conf, color in boxes_before:
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor=color, facecolor='none',
                                alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x + 5, y + 15, f'{conf:.2f}', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    ax1.set_title('Before NMS\n(5 overlapping detections)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Plot after NMS
    ax2.imshow(img)
    for x, y, w, h, conf, color in boxes_after:
        rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x + 5, y + 15, f'{conf:.2f}', fontsize=10,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    ax2.set_title('After NMS\n(Best detection kept)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.suptitle('Non-Maximum Suppression (NMS)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'nms_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_map_curve():
    """Create mAP (mean Average Precision) visualization"""
    np.random.seed(42)

    # Generate precision-recall curves for different classes
    recalls = np.linspace(0, 1, 100)

    classes = ['Person', 'Car', 'Dog', 'Cat', 'Bicycle']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#FFD93D']

    fig, ax = plt.subplots(figsize=(10, 7))

    aps = []
    for cls, color in zip(classes, colors):
        # Generate realistic PR curve
        precision = np.exp(-recalls * np.random.uniform(1.5, 3.0))
        precision = np.clip(precision + np.random.randn(100) * 0.05, 0, 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]  # Make monotonic

        ap = np.trapz(precision, recalls)
        aps.append(ap)

        ax.plot(recalls, precision, linewidth=2.5, label=f'{cls} (AP={ap:.3f})',
               color=color)

    mean_ap = np.mean(aps)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves (mAP = {mean_ap:.3f})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # Add mAP annotation
    ax.text(0.05, 0.15, f'mAP@0.5 = {mean_ap:.3f}',
           fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(output_dir / 'map_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Generating object detection figures...")
    create_detection_example()
    print("  ✓ Detection example")
    create_yolo_grid()
    print("  ✓ YOLO grid")
    create_rcnn_pipeline()
    print("  ✓ Faster R-CNN pipeline")
    create_nms_visualization()
    print("  ✓ NMS visualization")
    create_map_curve()
    print("  ✓ mAP curve")
    print("Done!")
