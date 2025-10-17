"""
Master script to generate all figures for Computer Vision & Deep Learning II
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import all figure generation modules
import image_classification
import segmentation_methods
import object_detection


def main():
    """Generate all figures for the module"""
    print("=" * 60)
    print("Generating All Figures for Computer Vision & Deep Learning II")
    print("=" * 60)
    print()

    # Generate image classification figures
    print("[1/3] Image Classification Figures (Enhanced)")
    print("-" * 60)
    image_classification.create_mnist_samples()
    print("  ✓ MNIST samples (PyTorch)")
    image_classification.create_cifar10_samples()
    print("  ✓ CIFAR-10 samples (PyTorch)")
    image_classification.create_confusion_matrix()
    print("  ✓ Confusion matrix")
    image_classification.create_training_curves()
    print("  ✓ Training curves")
    image_classification.create_cnn_architecture()
    print("  ✓ Advanced 3D CNN architecture")
    image_classification.create_feature_maps_visualization()
    print("  ✓ Feature map visualizations (scikit-image)")
    image_classification.create_multiclass_predictions()
    print("  ✓ Multi-class predictions with confidence")
    print()

    # Generate segmentation figures
    print("[2/3] Segmentation Figures")
    print("-" * 60)
    segmentation_methods.create_segmentation_types()
    print("  ✓ Segmentation types comparison")
    segmentation_methods.create_unet_architecture()
    print("  ✓ U-Net architecture")
    segmentation_methods.create_segmentation_example()
    print("  ✓ Segmentation example")
    segmentation_methods.create_iou_visualization()
    print("  ✓ IoU visualization")
    print()

    # Generate object detection figures
    print("[3/3] Object Detection Figures")
    print("-" * 60)
    object_detection.create_detection_example()
    print("  ✓ Detection example")
    object_detection.create_yolo_grid()
    print("  ✓ YOLO grid")
    object_detection.create_rcnn_pipeline()
    print("  ✓ Faster R-CNN pipeline")
    object_detection.create_nms_visualization()
    print("  ✓ NMS visualization")
    object_detection.create_map_curve()
    print("  ✓ mAP curve")
    print()

    print("=" * 60)
    print("✓ All enhanced figures generated successfully!")
    print("=" * 60)
    print()
    print(f"📁 Figures saved to: {scripts_dir.parent / 'figures'}")
    print()
    print("📊 Total figures generated:")
    figure_count = len(list((scripts_dir.parent / 'figures').glob('*.png')))
    print(f"   {figure_count} high-quality PNG visualizations")
    print()
    print("🎓 Data sources used:")
    print("   • PyTorch (torchvision): MNIST, CIFAR-10")
    print("   • scikit-image: Real photos, feature maps")
    print("   • Custom simulations: Segmentation, detection")
    print()


if __name__ == '__main__':
    main()
