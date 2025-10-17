# 🎯 Computer Vision & Deep Learning II

## 📖 Overview
This module covers **Deep Learning Applications in Computer Vision**, focusing on practical implementations and state-of-the-art architectures. Building on Part I's foundations, we explore advanced applications including image classification, segmentation, and object detection.

## 🎓 Learning Objectives
- Implement CNN architectures for image classification (MNIST, CIFAR-10)
- Understand and apply image segmentation techniques
- Master object detection algorithms (YOLO, Faster R-CNN)
- Evaluate model performance and optimize architectures
- Apply transfer learning to real-world problems

## 📁 Contents

### 📊 Slides (`slides/`)
- **File**: `deep_learning_applications_presentation.tex`
- **Compiled PDF**: `deep_learning_applications_presentation.pdf`
- **Coverage**: 50-60 slides covering:
  - Image classification architectures and datasets
  - Semantic and instance segmentation
  - Object detection frameworks (YOLO, R-CNN family)
  - Model evaluation and optimization
  - Real-world applications

### 🐍 Scripts (`scripts/`)
- `image_classification.py` - MNIST and CIFAR-10 implementations
- `segmentation_methods.py` - U-Net and segmentation visualizations
- `object_detection.py` - YOLO and Faster R-CNN demonstrations
- `generate_all_figures.py` - Master script for all figures

### 📓 Notebook (`notebooks/`)
- **File**: `deep_learning_applications_workshop.ipynb`
- **Duration**: 45-60 minutes
- **Interactive Activities**: Hands-on model training and evaluation
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/CMSC178IP/blob/main/10-ComputerVisionDeepLearningII/notebooks/deep_learning_applications_workshop.ipynb)

### 🖼️ Figures (`figures/`)
10-15 high-quality visualizations generated from scripts, including:
- Classification results on MNIST/CIFAR-10
- Segmentation masks and predictions
- Object detection bounding boxes
- Model architecture diagrams
- Performance comparisons

## 🚀 Quick Start

### Generate All Figures
```bash
cd scripts
python generate_all_figures.py
```

### Compile Slides
```bash
cd slides
pdflatex deep_learning_applications_presentation.tex
pdflatex deep_learning_applications_presentation.tex  # Second pass for references
```

### Run Jupyter Notebook
```bash
cd notebooks
jupyter notebook deep_learning_applications_workshop.ipynb
```

## 📦 Prerequisites

### Python Dependencies
```bash
pip install numpy matplotlib seaborn opencv-python scikit-image tensorflow torch torchvision pillow
```

### LaTeX Requirements
- TeX Live or MiKTeX distribution
- Metropolis Beamer theme
- Standard packages: amsmath, graphicx, hyperref

## 🎯 Key Topics

### 1. Image Classification
- **Datasets**: MNIST (handwritten digits), CIFAR-10 (natural images)
- **Architectures**: LeNet, AlexNet, VGG, ResNet
- **Training**: Data augmentation, regularization, optimization
- **Evaluation**: Accuracy, confusion matrices, per-class metrics

### 2. Image Segmentation
- **Semantic Segmentation**: Pixel-wise classification
- **Instance Segmentation**: Object-level segmentation
- **Architectures**: FCN, U-Net, Mask R-CNN
- **Applications**: Medical imaging, autonomous driving

### 3. Object Detection
- **Two-Stage Detectors**: R-CNN, Fast R-CNN, Faster R-CNN
- **One-Stage Detectors**: YOLO (You Only Look Once), SSD
- **Evaluation Metrics**: IoU, mAP, precision-recall curves
- **Real-time Applications**: Video surveillance, robotics

## 💡 Best Practices

### Model Training
- Start with pretrained models (transfer learning)
- Use appropriate data augmentation
- Monitor validation metrics to prevent overfitting
- Implement early stopping and learning rate scheduling

### Architecture Selection
- Match model complexity to dataset size
- Consider inference speed requirements
- Balance accuracy vs computational cost
- Use ensemble methods for critical applications

### Debugging
- Visualize predictions on validation data
- Check data preprocessing pipeline
- Monitor training curves (loss, accuracy)
- Test on edge cases and failure modes

## 🔍 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
**Solution**: Reduce batch size, use gradient accumulation, or mixed precision training

**Issue**: Model not converging
**Solution**: Check learning rate, verify data preprocessing, ensure proper initialization

**Issue**: Poor generalization
**Solution**: Add regularization (dropout, weight decay), increase training data, use data augmentation

**Issue**: Slow inference
**Solution**: Use model quantization, prune weights, switch to lighter architecture

## 📚 Additional Resources

### Papers
- ImageNet Classification with Deep CNNs (AlexNet) - Krizhevsky et al., 2012
- U-Net: Convolutional Networks for Biomedical Image Segmentation - Ronneberger et al., 2015
- You Only Look Once: Unified, Real-Time Object Detection - Redmon et al., 2016
- Faster R-CNN: Towards Real-Time Object Detection - Ren et al., 2015

### Tutorials
- PyTorch Image Classification Tutorial
- TensorFlow Object Detection API
- Fast.ai Practical Deep Learning Course
- Papers with Code - Computer Vision Benchmarks

### Datasets
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
- COCO (Detection/Segmentation): https://cocodataset.org/
- Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/

## 🎓 Course Integration
This module builds upon:
- **Module 09**: Deep learning fundamentals, CNN architectures, training basics
- **Module 08**: Feature extraction and traditional computer vision

This module prepares for:
- Advanced topics in computer vision
- Real-world project implementations
- Research in deep learning applications

## 👨‍🏫 Instructor Notes
- Emphasize practical implementation over theoretical details
- Encourage students to experiment with hyperparameters
- Provide access to GPU resources for training exercises
- Use pre-trained models to reduce training time
- Focus on model evaluation and interpretation

---

**Course**: CMSC 178IP - Digital Image Processing
**Institution**: University of the Philippines - Cebu
**Department**: Computer Science
