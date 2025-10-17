# Generative Models - Usage Guide

## Quick Start

### Installation

1. **Navigate to the directory**:
   ```bash
   cd 11-GenerativeModels
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Workshop

**Interactive Jupyter Notebook**:
```bash
jupyter notebook workshop.ipynb
```

This will open an interactive notebook where students can:
- Train autoencoders, VAEs, and GANs
- Generate new images
- Explore latent spaces
- Compare different generative models

### Generating Figures

**Generate all figures**:
```bash
cd scripts
python generate_all_figures.py
```

**Or generate specific figure sets**:
```bash
python generative_core.py          # Core concepts (autoencoders, VAE, GAN basics)
python generative_advanced.py      # Advanced topics (training dynamics, mode collapse)
python generative_applications.py  # Real-world applications (with Picsum images)
```

## What's New: Picsum Integration

The application figures now use **real images from Picsum.photos** for:

### Style Transfer Demonstrations
Shows how style transfer works with realistic photographs instead of synthetic data.

**Example usage in code**:
```python
from generative_applications import fetch_picsum_image

# Fetch a specific image (reproducible)
content_img = fetch_picsum_image(image_id=1015, width=256, height=256, grayscale=True)

# Fetch a random image
random_img = fetch_picsum_image(width=300, height=300)
```

### Image-to-Image Translation
Demonstrates domain transfer (e.g., color to grayscale, day to night) using diverse real photographs.

**Features**:
- Automatic fallback if no internet connection
- Reproducible results using specific image IDs
- Professional, educational-quality visualizations

## Workshop Activities

### Part 1: Autoencoders (30 minutes)
- Understand encoder-decoder architecture
- Implement and train basic autoencoder
- Visualize reconstructions
- **Key Learning**: Compression and reconstruction

### Part 2: Variational Autoencoders (30 minutes)
- Learn VAE loss function (reconstruction + KL divergence)
- Implement reparameterization trick
- Generate new samples from latent space
- Explore latent space structure
- **Key Learning**: Probabilistic generation

### Part 3: Generative Adversarial Networks (40 minutes)
- Build generator and discriminator
- Implement adversarial training loop
- Generate synthetic digits
- Compare with VAE results
- **Key Learning**: Adversarial training paradigm

## Files Overview

```
11-GenerativeModels/
├── README.md                      # Comprehensive documentation
├── PICSUM_INTEGRATION.md          # Details on Picsum usage
├── USAGE_GUIDE.md                 # This file
├── workshop.ipynb                 # Interactive student activities
├── requirements.txt               # Python dependencies
├── venv/                          # Virtual environment (created)
├── figures/                       # Generated visualizations
│   ├── autoencoder_architecture.png
│   ├── vae_architecture.png
│   ├── gan_architecture.png
│   ├── style_transfer_concept.png        # Uses Picsum!
│   ├── image_to_image_translation.png    # Uses Picsum! (NEW)
│   └── ... (18 more figures)
└── scripts/                       # Figure generation scripts
    ├── generative_core.py
    ├── generative_advanced.py
    ├── generative_applications.py # Updated with Picsum
    └── generate_all_figures.py
```

## Requirements

### Software
- Python 3.8+
- Jupyter Notebook
- Internet connection (optional, for Picsum images)

### Python Packages
- PyTorch (deep learning)
- torchvision (datasets)
- matplotlib (visualization)
- scikit-learn (PCA, metrics)
- scikit-image (image processing)
- requests (Picsum API)
- seaborn (enhanced plots)

### Hardware
- CPU: Any modern processor
- RAM: 4GB+ recommended
- GPU: Optional (for faster training)

## Tips for Instructors

1. **Pre-class Setup**:
   - Test the environment on student machines
   - Pre-download MNIST dataset
   - Generate all figures beforehand

2. **During Workshop**:
   - Start with autoencoder visualization
   - Emphasize the difference between AE and VAE
   - Show mode collapse in GAN training
   - Use the Picsum-based figures for real-world context

3. **Extension Ideas**:
   - Try different datasets (Fashion-MNIST, CIFAR-10)
   - Implement DCGAN or WGAN
   - Create conditional generation models
   - Experiment with latent space interpolation

## Troubleshooting

### Picsum Images Not Loading
**Symptom**: Warning messages about Picsum timeouts

**Solution**: This is normal! The script automatically uses fallback images from scikit-image. The figures will still be generated successfully.

### Out of Memory Errors
**Symptom**: CUDA out of memory or system memory errors

**Solution**:
- Reduce batch size in workshop.ipynb
- Use smaller latent dimensions
- Train on CPU instead of GPU

### Slow Training
**Symptom**: Training takes very long

**Solution**:
- Reduce number of epochs
- Use smaller network architectures
- Enable GPU if available (requires CUDA setup)

## Learning Outcomes

After completing this workshop, students will be able to:

✓ Explain the difference between discriminative and generative models
✓ Understand autoencoder architecture and limitations
✓ Implement VAE with proper loss function
✓ Build and train basic GANs
✓ Generate new images using trained models
✓ Compare and contrast different generative approaches
✓ Apply generative models to real-world problems

## Additional Resources

- **Papers with Code**: https://paperswithcode.com/task/image-generation
- **PyTorch GAN Tutorial**: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- **VAE Tutorial**: https://arxiv.org/abs/1606.05908
- **Picsum API**: https://picsum.photos
- **GAN Lab (Interactive)**: https://poloclub.github.io/ganlab/

## Assessment Ideas

1. **Formative**:
   - Complete workshop activities
   - Answer reflection questions
   - Generate samples from models

2. **Summative**:
   - Train GAN on new dataset
   - Implement GAN variant (DCGAN, WGAN)
   - Compare VAE vs GAN quantitatively
   - Write report on generation quality

## Contact

For questions or issues with this material, please contact the course instructor or teaching assistants.

---

**Version**: 1.1 (with Picsum integration)
**Last Updated**: October 2025
**Course**: CMSC 178 - Introduction to Image Processing
