# 🔧 Image Restoration and Geometric Processing

**CMSC 178IP - Digital Image Processing**
**Author:** Noel Jeffrey Pinton
**Institution:** University of the Philippines - Cebu
**Department:** Computer Science

---

## 🎯 Overview

This comprehensive educational package covers **Image Restoration and Geometric Processing** techniques in digital image processing. Students will learn fundamental and advanced methods for recovering original images from degraded versions and performing spatial transformations.

## 📚 Learning Objectives

By completing this module, students will be able to:

1. **Understand** the fundamental degradation model: $g(x,y) = h(x,y) * f(x,y) + \eta(x,y)$
2. **Apply** spatial domain restoration techniques (mean, Gaussian, median, bilateral filtering)
3. **Implement** frequency domain restoration methods (Wiener filter, Richardson-Lucy)
4. **Perform** geometric transformations and image registration
5. **Evaluate** restoration quality using PSNR and SSIM metrics
6. **Solve** real-world restoration problems in medical imaging, document processing, and computer vision

---

## 📁 Directory Structure

```
06 - Image Restoration and Geometric Processing/
├── 📊 figures/                    # Generated visualizations (13 PNG files)
│   ├── advanced_geometric_transforms.png
│   ├── computer_vision_preprocessing.png
│   ├── document_processing.png
│   ├── frequency_domain_restoration.png
│   ├── geometric_transformations.png
│   ├── image_inpainting.png
│   ├── image_registration.png
│   ├── interpolation_methods.png
│   ├── medical_image_restoration.png
│   ├── motion_blur_restoration.png
│   ├── noise_models.png
│   ├── quality_metrics.png
│   └── spatial_filtering.png
├── 📓 notebooks/                  # Interactive Jupyter workshop
│   └── image_restoration_workshop.ipynb
├── 🐍 scripts/                   # Python figure generation
│   ├── core_methods.py
│   ├── advanced_techniques.py
│   ├── real_world_examples.py
│   └── generate_all_figures.py
├── 📊 slides/                     # LaTeX Beamer presentation
│   ├── image_restoration_presentation.tex
│   └── image_restoration_presentation.pdf
└── 📖 README.md                   # This documentation
```

---

## 🚀 Quick Start

### 1. 📊 View the Presentation
```bash
# Navigate to slides directory
cd "06 - Image Restoration and Geometric Processing/slides"

# Open the PDF presentation
open image_restoration_presentation.pdf   # macOS
xdg-open image_restoration_presentation.pdf   # Linux
```

### 2. 🔬 Run the Interactive Workshop
```bash
# Open Jupyter notebook
jupyter notebook notebooks/image_restoration_workshop.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/image_restoration_workshop.ipynb
```

### 3. 🖼️ Generate All Figures
```bash
# Navigate to scripts directory
cd "06 - Image Restoration and Geometric Processing/scripts"

# Run the master generation script
python generate_all_figures.py
```

---

## 🛠️ Installation & Dependencies

### Required Python Packages
```bash
pip install numpy matplotlib scipy scikit-image opencv-python
```

### Detailed Dependencies
- **numpy** (≥1.19.0): Numerical computing
- **matplotlib** (≥3.3.0): Plotting and visualization
- **scipy** (≥1.5.0): Scientific computing and signal processing
- **scikit-image** (≥0.17.0): Image processing algorithms
- **opencv-python** (≥4.5.0): Computer vision operations

### For LaTeX Presentation
- **LaTeX distribution** (TeX Live, MiKTeX, or MacTeX)
- **Beamer class** with metropolis theme
- **Required LaTeX packages**: amsmath, amsfonts, amssymb, graphicx, tikz, tcolorbox

---

## 📊 Component Descriptions

### 🐍 Python Scripts (`scripts/`)

#### `core_methods.py`
Demonstrates fundamental restoration techniques:
- **Noise models**: Gaussian, salt & pepper, speckle
- **Spatial filtering**: Mean, Gaussian, median, bilateral
- **Geometric transformations**: Translation, rotation, scaling, shearing
- **Interpolation methods**: Nearest neighbor, bilinear, bicubic
- **Quality metrics**: PSNR and SSIM calculations

#### `advanced_techniques.py`
Advanced restoration and processing methods:
- **Motion blur restoration**: Wiener filter, Richardson-Lucy deconvolution
- **Image inpainting**: Biharmonic, Telea, and Navier-Stokes methods
- **Advanced geometric transforms**: Polar, log-polar, barrel distortion, elastic deformation
- **Frequency domain restoration**: Inverse filtering, constrained least squares

#### `real_world_examples.py`
Practical applications and processing pipelines:
- **Document processing**: Deskewing, noise reduction, binarization
- **Medical image restoration**: Anisotropic diffusion, artifact removal
- **Computer vision preprocessing**: Illumination correction, object detection
- **Image registration**: Feature-based alignment, quality assessment

#### `generate_all_figures.py`
Master script that:
- Executes all figure generation scripts
- Provides dependency checking
- Tracks execution time and success status
- Generates comprehensive summary report

### 📊 LaTeX Presentation (`slides/`)

**40+ slides** covering:
1. **Introduction** to image restoration concepts
2. **Degradation models** and noise characteristics
3. **Spatial domain methods** with mathematical formulations
4. **Frequency domain techniques** including Wiener and CLS filtering
5. **Motion blur** modeling and deblurring algorithms
6. **Image inpainting** methods and applications
7. **Geometric processing** and transformation hierarchies
8. **Quality assessment** metrics and evaluation
9. **Real-world applications** in various domains
10. **Best practices** and common pitfalls

### 📓 Interactive Notebook (`notebooks/`)

**45-60 minute workshop** featuring:
- **Hands-on coding** with guided examples
- **Progressive difficulty** from basic to advanced concepts
- **Student activity**: 15-minute restoration challenge
- **Quality assessment** with immediate feedback
- **Complete solutions** with detailed explanations
- **Google Colab compatibility** for easy access

### 📊 Generated Figures (`figures/`)

**13 high-quality visualizations** including:
- Noise model comparisons and characteristics
- Spatial filtering method demonstrations
- Frequency domain restoration results
- Geometric transformation examples
- Real-world application pipelines
- Quality metric evaluations

---

## 🎓 Usage Instructions

### For Instructors

#### 📊 Presentation Delivery
1. **Setup**: Ensure LaTeX environment with metropolis theme
2. **Compilation**: Run `pdflatex image_restoration_presentation.tex`
3. **Duration**: 50-minute lecture with Q&A time
4. **Figures**: All referenced images are automatically generated

#### 🔬 Workshop Facilitation
1. **Preparation**: Test notebook execution in target environment
2. **Student Setup**: Provide dependency installation instructions
3. **Activity Monitoring**: Circulate during 15-minute challenge
4. **Discussion**: Use solution reveals for learning reinforcement

#### 📊 Figure Integration
1. **Generation**: Run `generate_all_figures.py` before class
2. **Verification**: Check all 13 figures are created successfully
3. **Customization**: Modify scripts for specific examples if needed

### For Students

#### 🔍 Self-Study Path
1. **Start** with presentation slides for theoretical foundation
2. **Practice** with interactive notebook exercises
3. **Experiment** with parameter variations in restoration algorithms
4. **Challenge** yourself with the practical restoration task
5. **Explore** advanced techniques in real-world applications

#### 💻 Programming Practice
1. **Understand** each code block before executing
2. **Modify** parameters to see effects on restoration quality
3. **Implement** variations of demonstrated algorithms
4. **Measure** performance using provided quality metrics
5. **Document** observations and insights

---

## ⚙️ Build Instructions

### 📊 Regenerate All Figures
```bash
cd "06 - Image Restoration and Geometric Processing/scripts"
python generate_all_figures.py
```

**Expected output:**
- 13 PNG files in `../figures/` directory
- Execution summary with timing information
- Success/failure status for each script

### 📊 Rebuild LaTeX Presentation
```bash
cd "06 - Image Restoration and Geometric Processing/slides"
pdflatex image_restoration_presentation.tex
pdflatex image_restoration_presentation.tex  # Second run for references
```

**Note:** Ensure figures are generated before LaTeX compilation.

### 🔬 Test Notebook Execution
```bash
# Install nbconvert if not available
pip install nbconvert

# Test notebook execution
jupyter nbconvert --to notebook --execute \
    notebooks/image_restoration_workshop.ipynb \
    --output test_execution.ipynb
```

---

## 🌟 Key Features

### 🎯 Educational Focus
- **Progressive complexity** from basic concepts to advanced applications
- **Mathematical rigor** with intuitive explanations
- **Hands-on practice** with immediate visual feedback
- **Real-world relevance** through practical examples

### 💻 Technical Excellence
- **Error-free execution** with comprehensive testing
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Google Colab support** for browser-based execution
- **Professional visualizations** with consistent styling

### 📊 Comprehensive Coverage
- **Multiple restoration approaches** (spatial, frequency, iterative)
- **Various degradation types** (blur, noise, geometric distortion)
- **Quality assessment methods** (PSNR, SSIM, visual evaluation)
- **Application domains** (medical, document, computer vision)

---

## 🔧 Troubleshooting

### Common Issues

#### LaTeX Compilation Errors
**Problem**: Missing packages or overfull boxes
**Solution**:
```bash
# Install missing packages
tlmgr install metropolis tcolorbox pgfopts

# Check log file for specific errors
cat image_restoration_presentation.log | grep -i error
```

#### Python Import Errors
**Problem**: Missing dependencies
**Solution**:
```bash
# Install all required packages
pip install numpy matplotlib scipy scikit-image opencv-python

# For Conda users
conda install numpy matplotlib scipy scikit-image opencv
```

#### Figure Generation Failures
**Problem**: Script execution errors
**Solution**:
```bash
# Check individual script execution
python core_methods.py
python advanced_techniques.py
python real_world_examples.py

# Review error messages and install missing dependencies
```

#### Notebook Execution Issues
**Problem**: Kernel crashes or import failures
**Solution**:
```bash
# Restart Jupyter kernel
# Check Python environment
python --version
pip list | grep -E "(numpy|matplotlib|scipy|skimage|cv2)"

# For Google Colab, use:
!pip install scikit-image
```

### Performance Optimization

#### For Large Images
- Reduce image sizes for faster processing
- Use subset of iterations for iterative algorithms
- Consider parallel processing for batch operations

#### For Memory Constraints
- Process images in tiles for very large datasets
- Use lower precision (float32 instead of float64) when appropriate
- Clear variables after use with `del` statement

---

## 📊 Quality Assurance

### Validation Checklist

#### ✅ Technical Validation
- [ ] All Python scripts execute without errors
- [ ] LaTeX presentation compiles successfully with no overfull warnings
- [ ] Jupyter notebook runs completely in clean environment
- [ ] All 13 figures generate correctly
- [ ] Mathematical formulations are accurate
- [ ] Code comments are clear and helpful

#### ✅ Educational Validation
- [ ] Learning objectives are met through content
- [ ] Progressive difficulty maintains student engagement
- [ ] Practical examples reinforce theoretical concepts
- [ ] Student activity provides appropriate challenge
- [ ] Solutions are correct and well-explained

#### ✅ Content Validation
- [ ] Figures display correctly in LaTeX
- [ ] Interactive elements function properly
- [ ] Quality metrics produce reasonable values
- [ ] Real-world examples are relevant and current
- [ ] Best practices reflect industry standards

---

## 🤝 Contributing

### Reporting Issues
- Use GitHub issues for bug reports
- Include system information and error messages
- Provide minimal reproducible examples

### Submitting Improvements
- Fork repository and create feature branch
- Follow existing code style and documentation
- Test changes across different environments
- Submit pull request with clear description

### Content Suggestions
- Propose new examples or applications
- Suggest additional quality metrics
- Recommend current research references
- Share classroom experience feedback

---

## 📚 Additional Resources

### 📖 Recommended Reading
- **Gonzalez & Woods**: "Digital Image Processing" (Chapters 5, 11-12)
- **Szeliski**: "Computer Vision: Algorithms and Applications" (Chapter 3)
- **Pratt**: "Digital Image Processing" (Chapters 13-14)

### 🔗 Online Resources
- [scikit-image documentation](https://scikit-image.org/docs/stable/)
- [OpenCV documentation](https://docs.opencv.org/)
- [Richardson-Lucy algorithm](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)
- [Wiener filter theory](https://en.wikipedia.org/wiki/Wiener_filter)

### 📊 Datasets for Practice
- [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [SIDD Dataset](https://www.eecs.yorku.ca/~kamel/sidd/)

### 🔬 Research Papers
- Richardson, W. H. (1972). "Bayesian-Based Iterative Method of Image Restoration"
- Lucy, L. B. (1974). "An iterative technique for the rectification of observed distributions"
- Tomasi, C. & Manduchi, R. (1998). "Bilateral filtering for gray and color images"

---

## 📝 License & Citation

### License
This educational material is provided under the MIT License. See LICENSE file for details.

### Citation
If you use this material in your course or research, please cite:

```bibtex
@misc{pinton2024imagerestoration,
  title={Image Restoration and Geometric Processing Educational Package},
  author={Pinton, Noel Jeffrey},
  year={2024},
  institution={University of the Philippines - Cebu},
  course={CMSC 178IP - Digital Image Processing}
}
```

---

## 📧 Contact Information

**Instructor:** Noel Jeffrey Pinton
**Institution:** University of the Philippines - Cebu
**Department:** Computer Science
**Course:** CMSC 178IP - Digital Image Processing

**Support:**
- Course questions: Through your learning management system
- Technical issues: GitHub repository issues
- Content feedback: Direct email or course evaluation

---

## 📈 Version History

### Version 1.0 (2024)
- Initial release with complete educational package
- 13 generated figures with comprehensive coverage
- 40+ slide presentation with mathematical formulations
- Interactive 45-60 minute workshop with student activity
- Comprehensive documentation and troubleshooting guide

### Planned Updates
- Additional deep learning restoration methods
- Extended real-world application examples
- Video processing restoration techniques
- Advanced geometric transformation methods

---

**🎉 Thank you for using this Image Restoration and Geometric Processing educational package! Happy learning! 🚀**