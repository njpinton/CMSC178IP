# 🧠 Computer Vision and Deep Learning Approaches I

> **CMSC 178IP - Digital Image Processing**
> University of the Philippines - Cebu
> Department of Computer Science

A comprehensive educational package introducing neural networks, deep learning fundamentals, and convolutional neural networks (CNNs) for computer vision applications.

---

## 📚 Learning Objectives

By the end of this module, students will be able to:

1. **Understand** the fundamentals of neural networks and deep learning
2. **Implement** perceptrons and multi-layer perceptrons from scratch
3. **Apply** various activation functions and understand their properties
4. **Explain** gradient descent, backpropagation, and optimization algorithms
5. **Build** convolutional neural networks for image classification
6. **Analyze** CNN architectures and understand convolution/pooling operations
7. **Train** deep neural networks with proper regularization techniques
8. **Evaluate** model performance using appropriate metrics
9. **Apply** data augmentation and transfer learning concepts
10. **Debug** common issues in deep learning pipelines

---

## 📁 Directory Structure

```
09-ComputerVisionDeepLearningI/
├── figures/                    # Generated visualizations (15 PNG files)
│   ├── 01_activation_functions.png
│   ├── 02_perceptron_diagram.png
│   ├── 03_mlp_architecture.png
│   ├── 04_loss_functions.png
│   ├── 05_gradient_descent.png
│   ├── 06_convolution_operation.png
│   ├── 07_pooling_operations.png
│   ├── 08_cnn_architecture.png
│   ├── 09_feature_maps.png
│   ├── 10_learning_curves.png
│   ├── 11_preprocessing_pipeline.png
│   ├── 12_data_augmentation.png
│   ├── 13_classification_example.png
│   ├── 14_confusion_matrix.png
│   └── 15_overfitting_example.png
├── notebooks/                  # Interactive Jupyter workshop
│   └── cv_deep_learning_workshop.ipynb
├── scripts/                    # Python scripts for figure generation
│   ├── core_methods.py
│   ├── advanced_techniques.py
│   ├── real_world_examples.py
│   └── generate_all_figures.py
├── slides/                     # LaTeX Beamer presentation
│   ├── cv_deep_learning_presentation.tex
│   └── cv_deep_learning_presentation.pdf
└── README.md                   # This file
```

---

## 🎯 Module Components

### 1. 📊 Presentation Slides

**File:** `slides/cv_deep_learning_presentation.tex` (PDF: `cv_deep_learning_presentation.pdf`)

**Content Coverage (66 slides):**
- Introduction to neural networks and deep learning
- Perceptrons and multi-layer perceptrons
- Activation functions (Sigmoid, ReLU, Tanh, Leaky ReLU)
- Loss functions (MSE, Cross-Entropy)
- Gradient descent and optimization
- Backpropagation algorithm
- Convolutional Neural Networks (CNNs)
- Convolution and pooling operations
- CNN architectures (LeNet, AlexNet, VGG, ResNet)
- Training deep networks
- Overfitting and regularization
- Data augmentation techniques
- Transfer learning
- Evaluation metrics and best practices

**Build Instructions:**
```bash
cd slides/
pdflatex cv_deep_learning_presentation.tex
pdflatex cv_deep_learning_presentation.tex  # Run twice for proper references
```

### 2. 🐍 Python Scripts

**Location:** `scripts/`

**Files:**
- **`core_methods.py`** - Neural network fundamentals
  - Activation function visualizations
  - Perceptron and MLP diagrams
  - Loss function comparisons
  - Gradient descent visualization

- **`advanced_techniques.py`** - CNN operations
  - Convolution operation demonstration
  - Pooling operations (max and average)
  - CNN architecture diagrams
  - Feature map visualizations
  - Learning curves

- **`real_world_examples.py`** - Practical applications
  - Image preprocessing pipeline
  - Data augmentation examples
  - Classification problem visualization
  - Confusion matrix
  - Overfitting vs good fit examples

- **`generate_all_figures.py`** - Master script to generate all figures

**Usage:**
```bash
cd scripts/
python3 generate_all_figures.py
```

**Dependencies:**
```bash
pip install numpy matplotlib seaborn opencv-python scipy pillow
```

### 3. 📓 Interactive Jupyter Notebook

**File:** `notebooks/cv_deep_learning_workshop.ipynb`

**Duration:** 45-60 minutes

**Structure:**
1. **Setup & Imports** - Environment preparation
2. **Part 1: Problem Understanding** - Motivation and dataset introduction
3. **Part 2: Core Methods** - Perceptrons, activations, forward propagation
4. **Part 3: Advanced Techniques** - CNN implementation and training
5. **Part 4: Diagnostic Tools** - Evaluation and visualization
6. **Part 5: Best Practices** - Overfitting, augmentation, tuning
7. **Part 6: Student Activity** - 15-minute hands-on challenge
8. **Part 7: Solutions** - Complete worked examples
9. **Part 8: Summary** - Key takeaways and next steps

**Dataset:** CIFAR-10 (10 classes, 60,000 images)

**Running Locally:**
```bash
cd notebooks/
jupyter notebook cv_deep_learning_workshop.ipynb
```

**Running on Google Colab:**
- Click the "Open in Colab" badge at the top of the notebook
- Or visit: [Your Colab Link Here]

**Requirements:**
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 4. 📊 Generated Figures

**Location:** `figures/`

All 15 high-quality PNG figures are automatically generated by the Python scripts and referenced in both the presentation and notebook.

---

## 🚀 Quick Start Guide

### Option 1: Complete Build (Recommended)

```bash
# 1. Generate all figures first
cd scripts/
python3 generate_all_figures.py

# 2. Build presentation
cd ../slides/
pdflatex cv_deep_learning_presentation.tex
pdflatex cv_deep_learning_presentation.tex

# 3. Run notebook
cd ../notebooks/
jupyter notebook cv_deep_learning_workshop.ipynb
```

### Option 2: Presentation Only

```bash
cd slides/
pdflatex cv_deep_learning_presentation.tex
open cv_deep_learning_presentation.pdf  # macOS
# or: xdg-open cv_deep_learning_presentation.pdf  # Linux
# or: start cv_deep_learning_presentation.pdf     # Windows
```

### Option 3: Interactive Workshop Only

```bash
cd notebooks/
jupyter notebook cv_deep_learning_workshop.ipynb
```

---

## 🔧 Prerequisites

### Software Requirements
- **Python 3.8+** with pip
- **Jupyter Notebook** or **JupyterLab**
- **LaTeX Distribution:**
  - macOS: MacTeX (`brew install --cask mactex`)
  - Linux: TeX Live (`sudo apt-get install texlive-full`)
  - Windows: MiKTeX or TeX Live

### Python Dependencies

**For Scripts:**
```bash
pip install numpy matplotlib seaborn opencv-python scipy pillow
```

**For Notebook:**
```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

**Or install all at once:**
```bash
pip install numpy matplotlib seaborn opencv-python scipy pillow tensorflow scikit-learn jupyter
```

### Hardware Recommendations
- **CPU:** Multi-core processor recommended
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** Optional but recommended for faster training (CUDA-compatible)
- **Storage:** 2GB free space for datasets and outputs

---

## 📖 Detailed Component Descriptions

### Activation Functions
Learn about and visualize:
- **Sigmoid:** Smooth, bounded [0,1], suffers from vanishing gradients
- **Tanh:** Zero-centered [-1,1], better than sigmoid
- **ReLU:** Fast, simple, most popular for hidden layers
- **Leaky ReLU:** Prevents dead neurons, small gradient for negative values

### Neural Network Architectures
- **Perceptron:** Single-layer linear classifier
- **MLP:** Multi-layer perceptron with hidden layers
- **CNN:** Convolutional networks for spatial data
  - Convolutional layers (feature extraction)
  - Pooling layers (downsampling)
  - Fully connected layers (classification)

### Training Techniques
- **Gradient Descent:** Batch, stochastic, mini-batch variants
- **Backpropagation:** Efficient gradient computation via chain rule
- **Optimization:** SGD, Momentum, RMSprop, Adam
- **Regularization:** L1/L2, dropout, batch normalization
- **Data Augmentation:** Rotation, flip, zoom, color jittering

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positives / Predicted positives
- **Recall:** True positives / Actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed breakdown of predictions

---

## 🎓 Teaching Notes

### Recommended Lecture Flow

1. **Introduction (10 min)**
   - Motivation for deep learning in computer vision
   - Historical context: Traditional CV → Deep Learning

2. **Neural Network Fundamentals (20 min)**
   - Perceptron model and learning
   - Activation functions
   - Multi-layer perceptrons
   - Forward and backward propagation

3. **Convolutional Neural Networks (25 min)**
   - Convolution operation
   - Feature maps and filters
   - Pooling layers
   - CNN architectures

4. **Training and Optimization (20 min)**
   - Loss functions
   - Gradient descent and variants
   - Regularization techniques
   - Hyperparameter tuning

5. **Hands-On Workshop (45-60 min)**
   - Work through Jupyter notebook
   - Student activity
   - Discussion and Q&A

### Student Activity Guidelines

The notebook includes a 15-minute student activity where students:
1. Build a CNN achieving >60% accuracy on CIFAR-10
2. Experiment with different architectures
3. Apply data augmentation
4. Evaluate and visualize results

**Hints provided:**
- Starter code with TODO markers
- Architectural suggestions
- Hyperparameter recommendations

**Solutions included** in separate section for instructor reference.

---

## 🐛 Troubleshooting

### Issue: Figures not found in LaTeX compilation

**Solution:**
```bash
cd scripts/
python3 generate_all_figures.py
cd ../slides/
pdflatex cv_deep_learning_presentation.tex
```

Ensure figures are generated before compiling LaTeX.

### Issue: ModuleNotFoundError for Python scripts

**Solution:**
```bash
pip install numpy matplotlib seaborn opencv-python scipy pillow
```

Or use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: TensorFlow installation problems

**Solution:**

For Apple Silicon (M1/M2) Macs:
```bash
pip install tensorflow-macos tensorflow-metal
```

For other systems:
```bash
pip install tensorflow
```

For GPU support (NVIDIA):
```bash
pip install tensorflow-gpu
```

### Issue: Jupyter kernel crashes during training

**Solution:**
- Reduce batch size in the notebook
- Use fewer epochs for initial testing
- Close other applications to free up RAM
- Consider using Google Colab for cloud resources

### Issue: LaTeX compilation warnings (overfull boxes)

**Note:** Some overfull vbox warnings may appear but don't affect PDF quality significantly. If content appears cut off:
- Check slide content isn't too dense
- Use smaller font sizes where appropriate
- Split dense slides into multiple slides

---

## 📚 Additional Resources

### Textbooks
- **Deep Learning** by Goodfellow, Bengio, and Courville
- **Neural Networks and Deep Learning** by Michael Nielsen (free online)
- **Hands-On Machine Learning** by Aurélien Géron

### Online Courses
- **CS231n:** Convolutional Neural Networks for Visual Recognition (Stanford)
- **fast.ai:** Practical Deep Learning for Coders
- **Deep Learning Specialization** by Andrew Ng (Coursera)

### Papers
- **LeNet-5:** LeCun et al., 1998
- **AlexNet:** Krizhevsky et al., 2012 (ImageNet Classification)
- **VGGNet:** Simonyan & Zisserman, 2014
- **ResNet:** He et al., 2015 (Deep Residual Learning)

### Frameworks
- **TensorFlow/Keras:** High-level, beginner-friendly
- **PyTorch:** Research-oriented, dynamic computation
- **JAX:** High-performance, functional programming

### Datasets
- **MNIST:** Handwritten digits (beginner)
- **CIFAR-10/100:** Natural images (intermediate)
- **ImageNet:** Large-scale image classification (advanced)
- **COCO:** Object detection and segmentation

---

## 🤝 Contributing

This is an educational package for CMSC 178IP. Suggestions for improvements:
- Additional visualizations
- More comprehensive examples
- Enhanced student activities
- Updated architectures and techniques

Contact: Noel Jeffrey Pinton (instructor)

---

## 📄 License

Educational material for CMSC 178IP - Digital Image Processing
University of the Philippines - Cebu
Department of Computer Science

---

## ✅ Quality Checklist

- [x] All Python scripts execute without errors
- [x] LaTeX presentation compiles to PDF successfully
- [x] Jupyter notebook runs completely in local environment
- [x] All 15 figures generated and referenced correctly
- [x] Mathematical formulations properly formatted
- [x] Student activity includes challenge and solution
- [x] Cross-platform compatibility (macOS, Linux, Windows)
- [x] All external dependencies documented
- [x] Colab compatibility for cloud execution
- [x] Professional styling and formatting
- [x] Comprehensive documentation and instructions

---

## 🎯 Learning Outcomes Assessment

Students should be able to demonstrate:

1. **Knowledge:** Explain neural network fundamentals and CNN architectures
2. **Comprehension:** Understand how gradient descent and backpropagation work
3. **Application:** Implement CNNs for image classification tasks
4. **Analysis:** Evaluate model performance using appropriate metrics
5. **Synthesis:** Design CNN architectures for specific problems
6. **Evaluation:** Debug and improve deep learning models

---

## 📞 Support

For questions or issues:
- Email: instructor@example.edu
- Office Hours: [Schedule]
- Course Forum: [Link]

---

**Last Updated:** October 2025
**Version:** 1.0
**Module:** 09 - Computer Vision and Deep Learning Approaches I
