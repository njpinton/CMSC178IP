# 📊 Image Segmentation and Morphological Processing

**CMSC 178IP - Digital Image Processing**
**Topic 08: Segmentation and Morphology**

## 🎯 Learning Objectives

By completing this module, students will be able to:

1. **Understand and apply segmentation methods:**
   - Global, local, and Otsu's thresholding techniques
   - Region-based segmentation (region growing, watershed)
   - Choose appropriate methods for different scenarios

2. **Master morphological operations:**
   - Binary morphology (erosion, dilation, opening, closing)
   - Grayscale morphology and its applications
   - Structuring element design and selection

3. **Combine techniques for practical applications:**
   - Noise removal pipelines
   - Object extraction and counting
   - Edge detection using morphology
   - Multi-scale segmentation approaches

## 📁 Directory Structure

```
08-SegmentationMorphology/
├── figures/              # Generated visualizations (15 PNG files)
│   ├── 01_global_thresholding.png
│   ├── 02_otsu_thresholding.png
│   ├── 03_local_thresholding.png
│   ├── 04_region_growing.png
│   ├── 05_watershed_segmentation.png
│   ├── 06_binary_erosion_dilation.png
│   ├── 07_binary_opening_closing.png
│   ├── 08_structuring_elements.png
│   ├── 09_morphological_gradient.png
│   ├── 10_grayscale_morphology.png
│   ├── 11_noise_removal_pipeline.png
│   ├── 12_edge_detection_comparison.png
│   ├── 13_object_extraction.png
│   ├── 14_texture_enhancement.png
│   └── 15_multi_scale_segmentation.png
├── notebooks/            # Interactive Jupyter workshop
│   └── segmentation_morphology_workshop.ipynb
├── scripts/              # Python scripts for figure generation
│   ├── segmentation_methods.py
│   ├── morphological_operations.py
│   ├── advanced_applications.py
│   └── generate_all_figures.py
├── slides/               # LaTeX Beamer presentation
│   ├── segmentation_morphology_presentation.tex
│   └── segmentation_morphology_presentation.pdf
└── README.md            # This file

```

## 🚀 Quick Start

### Prerequisites

**Required Python packages:**
```bash
numpy
matplotlib
scikit-image
scipy
```

**LaTeX requirements:**
- pdflatex
- beamer class
- metropolis theme

### Installation

```bash
# Install Python dependencies
pip install numpy matplotlib scikit-image scipy

# For LaTeX (macOS with MacTeX)
# Ensure you have a complete LaTeX distribution installed
```

## 📚 Component Descriptions

### 1. Figures (`figures/`)

High-quality visualizations demonstrating:

**Segmentation Methods (Figures 1-5):**
- Global thresholding with different threshold values
- Otsu's automatic thresholding method
- Local (adaptive) thresholding for varying illumination
- Region growing segmentation
- Watershed segmentation pipeline

**Morphological Operations (Figures 6-10):**
- Binary erosion and dilation effects
- Opening and closing for noise removal
- Different structuring element shapes
- Morphological gradient for edge detection
- Grayscale morphology operations and top-hat transforms

**Advanced Applications (Figures 11-15):**
- Complete noise removal pipeline
- Edge detection method comparison
- Object extraction and analysis workflow
- Texture enhancement using top-hat transforms
- Multi-scale segmentation approaches

### 2. Python Scripts (`scripts/`)

#### `segmentation_methods.py`
Generates figures demonstrating thresholding and region-based segmentation:
- Global thresholding with multiple thresholds
- Otsu's automatic threshold selection
- Local (adaptive) thresholding
- Simple region growing implementation
- Watershed segmentation with distance transform

#### `morphological_operations.py`
Demonstrates binary and grayscale morphology:
- Binary erosion and dilation with different SE sizes
- Opening and closing for noise removal
- Various structuring element shapes
- Morphological gradient (internal, external, full)
- Grayscale operations and top-hat transforms

#### `advanced_applications.py`
Shows practical applications combining techniques:
- Morphological noise removal pipeline
- Edge detection method comparison
- Object extraction and property analysis
- Texture enhancement at multiple scales
- Multi-scale segmentation strategies

#### `generate_all_figures.py`
Master script that executes all visualization scripts in sequence.

**Usage:**
```bash
cd scripts
python generate_all_figures.py
```

### 3. Jupyter Notebook (`notebooks/`)

**Interactive workshop:** `segmentation_morphology_workshop.ipynb`

**Duration:** 45-60 minutes including activities

**Structure:**
1. **Setup & Imports** - Environment preparation
2. **Part 1: Thresholding Methods** - Global, Otsu's, and local thresholding
3. **Part 2: Region-Based Segmentation** - Watershed algorithm
4. **Part 3: Binary Morphology** - Erosion, dilation, opening, closing
5. **Part 4: Grayscale Morphology** - Operations on intensity images
6. **Part 5: Student Activity** - Coin counting and analysis challenge
7. **Summary** - Key takeaways and next steps

**Google Colab:**
The notebook includes a Colab badge for easy cloud execution.

**Run locally:**
```bash
cd notebooks
jupyter notebook segmentation_morphology_workshop.ipynb
```

### 4. Presentation Slides (`slides/`)

**LaTeX Beamer presentation:** 59 slides covering comprehensive theory and applications

**Topics covered:**
- Introduction to image segmentation
- Thresholding methods (global, Otsu's, local)
- Region-based segmentation (region growing, watershed)
- Mathematical morphology foundations
- Binary morphological operations
- Grayscale morphology
- Advanced applications
- Best practices and guidelines

**Build instructions:**
```bash
cd slides
pdflatex segmentation_morphology_presentation.tex
pdflatex segmentation_morphology_presentation.tex  # Run twice for references
```

The PDF is also included for immediate use.

## 🔬 Key Concepts Covered

### Segmentation Methods

1. **Global Thresholding**
   - Simple intensity-based partitioning
   - Binary segmentation: `g(x,y) = 1 if f(x,y) > T, else 0`
   - Best for uniform illumination

2. **Otsu's Method**
   - Automatic threshold selection
   - Maximizes between-class variance
   - Optimal for bimodal histograms

3. **Local Thresholding**
   - Adaptive to local neighborhoods
   - Handles varying illumination
   - Block-based threshold computation

4. **Region Growing**
   - Seed-based region expansion
   - Similarity criterion-driven
   - Produces connected regions

5. **Watershed Segmentation**
   - Topographic interpretation
   - Marker-controlled approach
   - Effective for separating touching objects

### Morphological Operations

1. **Binary Erosion:** `A ⊖ B = {z | B_z ⊆ A}`
   - Shrinks objects
   - Removes small features
   - Separates touching objects

2. **Binary Dilation:** `A ⊕ B = {z | B̂_z ∩ A ≠ ∅}`
   - Expands objects
   - Fills small holes
   - Connects nearby objects

3. **Opening:** `A ∘ B = (A ⊖ B) ⊕ B`
   - Removes small bright regions
   - Smooths contours
   - Idempotent operation

4. **Closing:** `A • B = (A ⊕ B) ⊖ B`
   - Fills small dark regions
   - Smooths contours
   - Idempotent operation

5. **Grayscale Operations**
   - Erosion: local minimum filter
   - Dilation: local maximum filter
   - Top-hat transforms for feature extraction

## 💡 Practical Applications

### Object Counting
Combine thresholding, morphology, and watershed to:
1. Segment objects from background
2. Clean noise with opening/closing
3. Separate touching objects
4. Count and analyze regions

### Noise Removal
Morphological pipeline:
1. Opening → removes salt noise
2. Closing → removes pepper noise
3. Remove small objects/holes
4. Result: clean binary image

### Edge Detection
Morphological gradient:
- Full gradient: `Dilation - Erosion`
- Internal: `Original - Erosion`
- External: `Dilation - Original`

### Image Enhancement
Top-hat transforms:
- White top-hat: extract bright features
- Black top-hat: extract dark features
- Enhanced = Original + WTH - BTH

## 🎓 Teaching Notes

### Lecture Flow (90 minutes)

1. **Introduction (10 min)**
   - Motivation and applications
   - Segmentation challenges

2. **Thresholding (20 min)**
   - Global methods
   - Otsu's algorithm derivation
   - Local thresholding

3. **Region-Based Methods (15 min)**
   - Region growing concept
   - Watershed algorithm
   - Marker-controlled watershed

4. **Binary Morphology (20 min)**
   - Erosion and dilation
   - Opening and closing
   - Structuring elements

5. **Grayscale Morphology (15 min)**
   - Extension to grayscale
   - Top-hat transforms
   - Morphological gradient

6. **Applications (10 min)**
   - Real-world examples
   - Combining techniques

### Workshop Activities

**Activity 1: Threshold Comparison (10 min)**
- Compare global vs. local thresholding
- Analyze histogram characteristics

**Activity 2: Morphological Noise Removal (10 min)**
- Apply opening and closing
- Tune structuring element size

**Activity 3: Object Counting (15 min)**
- Complete segmentation pipeline
- Analyze region properties

## ⚙️ Troubleshooting

### Common Issues

**Issue:** Figures not displaying in notebook
**Solution:** Ensure matplotlib backend is set correctly
```python
%matplotlib inline
```

**Issue:** LaTeX compilation fails
**Solution:** Ensure all required packages are installed
```bash
# macOS
sudo tlmgr install metropolis beamertheme-metropolis

# Ubuntu
sudo apt-get install texlive-latex-extra texlive-fonts-extra
```

**Issue:** Import errors in Python scripts
**Solution:** Install missing packages
```bash
pip install -r requirements.txt
```

## 📖 Additional Resources

### Textbooks
- Gonzalez & Woods: *Digital Image Processing* (Chapter 10)
- Soille: *Morphological Image Analysis*
- Serra: *Image Analysis and Mathematical Morphology*

### Online Resources
- scikit-image documentation: https://scikit-image.org/
- Morphology tutorials: https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm
- Watershed algorithm: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html

### Modern Approaches
- Deep learning segmentation (U-Net, Mask R-CNN)
- Superpixel methods (SLIC, Felzenszwalb)
- Graph-based segmentation

## 🔍 Assessment Suggestions

### Conceptual Understanding
1. Explain when to use global vs. local thresholding
2. Describe the difference between opening and closing
3. Explain how watershed separates touching objects

### Practical Skills
1. Implement Otsu's method from scratch
2. Design structuring elements for specific tasks
3. Build a complete object counting pipeline

### Analysis
1. Compare segmentation methods on challenging images
2. Analyze the effect of SE size on morphological operations
3. Evaluate segmentation quality metrics

## 📝 License and Attribution

**Author:** Noel Jeffrey Pinton
**Institution:** University of the Philippines - Cebu
**Course:** CMSC 178IP - Digital Image Processing

This educational material is created for academic purposes.

## 🤝 Contributing

If you find errors or have suggestions for improvement:
1. Document the issue clearly
2. Provide specific examples
3. Suggest concrete solutions

## 📧 Contact

For questions or feedback about this module, please contact the course instructor.

---

**Last Updated:** October 2025
**Version:** 1.0

🎉 **Happy Learning!** Segmentation and morphology are fundamental tools in computer vision and image analysis.
