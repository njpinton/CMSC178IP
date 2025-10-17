# Lecture 11: Generative Models

## Overview

This lecture introduces generative models in computer vision, covering autoencoders, variational autoencoders (VAEs), and generative adversarial networks (GANs). Students will learn how these models can learn to generate new images and understand the underlying principles of each approach.

## Learning Objectives

By the end of this lecture, students will be able to:

1. Understand the architecture and purpose of autoencoders
2. Explain the difference between autoencoders and variational autoencoders
3. Understand the adversarial training paradigm of GANs
4. Implement basic versions of these models using PyTorch
5. Compare and contrast different generative modeling approaches
6. Apply generative models to practical computer vision tasks

## Contents

### 1. Presentation (`lecture11_generative_models.pdf`)

A comprehensive Beamer presentation covering:

- **Introduction to Generative Modeling**
  - What are generative models?
  - Applications in computer vision
  - Overview of different approaches

- **Autoencoders**
  - Architecture: encoder, latent space, decoder
  - Training objective and loss functions
  - Applications: compression, denoising, anomaly detection
  - Limitations for generation

- **Variational Autoencoders (VAEs)**
  - Probabilistic latent spaces
  - Reparameterization trick
  - VAE loss function: reconstruction + KL divergence
  - Generating new samples
  - Latent space exploration

- **Generative Adversarial Networks (GANs)**
  - Adversarial training paradigm
  - Generator and discriminator architecture
  - Training dynamics and challenges
  - GAN variants: DCGAN, WGAN, StyleGAN
  - Mode collapse and training stability

- **Advanced Topics**
  - Conditional generation
  - Image-to-image translation (pix2pix, CycleGAN)
  - Diffusion models (brief overview)
  - Evaluation metrics: FID, IS

- **Applications**
  - Image synthesis and editing
  - Data augmentation
  - Super-resolution
  - Style transfer
  - Anomaly detection

### 2. Interactive Workshop (`workshop.ipynb`)

A hands-on Jupyter notebook with guided activities:

**Part 1: Autoencoders**
- Understand encoder-decoder architecture
- Complete autoencoder implementation
- Train on MNIST dataset
- Visualize reconstructions

**Part 2: Variational Autoencoders**
- Understand VAE loss function (reconstruction + KL divergence)
- Implement reparameterization trick
- Train VAE on MNIST
- Generate new samples
- Explore latent space structure

**Part 3: Generative Adversarial Networks**
- Build generator and discriminator
- Implement adversarial training loop
- Train GAN on MNIST
- Generate synthetic digits
- Compare with VAE results

**Activities Include:**
- Completing model architectures
- Understanding loss functions
- Analyzing latent space organization
- Generating new images
- Comparing different generative approaches

**Reflection Questions:**
- Differences between autoencoders and VAEs
- Why VAEs are better for generation
- Challenges in GAN training
- Quality comparison across methods
- Real-world applications

**Extension Activities:**
- Latent space interpolation
- Conditional generation
- Denoising autoencoders
- Architecture experiments
- GAN improvements (WGAN, DCGAN)

### 3. Figures (`figures/`)

All visualizations used in the presentation, organized into three categories:

#### Core Concepts (`core/`)
- `autoencoder_architecture.png` - Basic autoencoder structure
- `vae_architecture.png` - VAE with probabilistic latent space
- `gan_architecture.png` - GAN training paradigm
- `latent_space_comparison.png` - AE vs VAE latent spaces
- `vae_reparameterization.png` - Reparameterization trick
- `gan_training_dynamics.png` - Generator vs discriminator loss

#### Advanced Topics (`advanced/`)
- `dcgan_architecture.png` - Deep convolutional GAN
- `conditional_generation.png` - Class-conditional generation
- `gan_mode_collapse.png` - Mode collapse visualization
- `vae_interpolation.png` - Latent space interpolation
- `latent_space_exploration.png` - Latent traversal effects

#### Applications (`applications/`)
- `image_generation_comparison.png` - Different model outputs
- `anomaly_detection.png` - Using autoencoders for anomalies
- `data_augmentation.png` - Synthetic training data
- `image_editing.png` - Latent space manipulation
- `style_transfer.png` - Style and content separation

### 4. Figure Generation Scripts (`scripts/`)

Three Python scripts to regenerate all figures:

- `generate_core_figures.py` - Core concept visualizations
- `generate_advanced_figures.py` - Advanced technique visualizations
- `generate_application_figures.py` - Application examples (uses Picsum.photos for realistic images)

To regenerate all figures:
```bash
cd scripts
python generative_core.py
python generative_advanced.py
python generative_applications.py
```

Or use the convenience script:
```bash
python scripts/generate_all_figures.py
```

**Note**: Some figures use [Picsum.photos](https://picsum.photos) to fetch realistic placeholder images for demonstrating style transfer and image-to-image translation. An internet connection is required for these figures. If Picsum is unavailable, the scripts will fall back to using built-in scikit-image datasets.

## Prerequisites

### Knowledge Prerequisites
- Linear algebra (matrix operations, eigenvalues)
- Probability theory (distributions, KL divergence)
- Deep learning basics (neural networks, backpropagation)
- Convolutional neural networks (CNN architectures)
- Python programming

### Software Requirements

Install required packages:
```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch (≥1.9.0)
- torchvision (≥0.10.0)
- NumPy (≥1.19.0)
- Matplotlib (≥3.3.0)
- scikit-learn (≥0.24.0)
- Jupyter (≥1.0.0)

**Optional:**
- CUDA toolkit (for GPU acceleration)

## Usage

### For Instructors

1. **Lecture Delivery**:
   - Use `lecture11_generative_models.pdf` for the main presentation
   - Estimated time: 60-75 minutes
   - Includes discussion prompts and examples

2. **Workshop Session**:
   - Distribute `workshop.ipynb` to students
   - Estimated time: 90-120 minutes
   - Students work through guided activities
   - Includes reflection questions and extensions

3. **Customization**:
   - Modify scripts in `scripts/` to create custom figures
   - Edit `workshop.ipynb` to adjust difficulty or focus
   - Regenerate presentation with updated figures

### For Students

1. **Pre-lecture Preparation**:
   - Review deep learning basics
   - Ensure software environment is set up
   - Download MNIST dataset (happens automatically in workshop)

2. **During Lecture**:
   - Follow along with presentation
   - Take notes on key concepts
   - Ask questions about architectures and training

3. **Workshop Activities**:
   ```bash
   jupyter notebook workshop.ipynb
   ```
   - Complete the TODOs in the notebook
   - Run all cells and observe outputs
   - Answer reflection questions
   - Try extension activities if time permits

4. **Post-lecture**:
   - Review generated samples
   - Experiment with hyperparameters
   - Explore additional GAN variants
   - Read recommended papers

## Key Concepts

### Autoencoders
- **Encoder**: Maps input x to latent representation z
- **Decoder**: Reconstructs input from z
- **Loss**: Reconstruction error (e.g., MSE)
- **Limitation**: Latent space may have "holes" - not all points correspond to valid data

### Variational Autoencoders (VAEs)
- **Probabilistic**: Encoder outputs distribution parameters (μ, σ²)
- **Reparameterization**: z = μ + σ * ε, where ε ~ N(0, 1)
- **Loss**: Reconstruction loss + KL divergence
- **Advantage**: Smooth, continuous latent space - can sample new data

### Generative Adversarial Networks (GANs)
- **Generator G**: Creates fake samples from noise z
- **Discriminator D**: Distinguishes real from fake
- **Training**: Adversarial min-max game
  - D tries to maximize: E[log D(x)] + E[log(1 - D(G(z)))]
  - G tries to minimize: E[log(1 - D(G(z)))]
- **Challenges**: Mode collapse, training instability, vanishing gradients

## Comparison of Approaches

| Aspect | Autoencoder | VAE | GAN |
|--------|-------------|-----|-----|
| **Latent Space** | Deterministic | Probabilistic | N/A (noise input) |
| **Training** | Direct reconstruction | Reconstruction + KL | Adversarial |
| **Generation Quality** | Poor | Good | Excellent |
| **Training Stability** | Stable | Stable | Unstable |
| **Latent Interpolation** | Difficult | Smooth | N/A |
| **Best Use Case** | Compression, features | Controlled generation | High-quality synthesis |

## Common Issues and Solutions

### VAE Issues
1. **Blurry reconstructions**: Increase model capacity, try β-VAE
2. **Posterior collapse**: Reduce KL weight, use warm-up schedule
3. **Poor generations**: Increase latent dimension, train longer

### GAN Issues
1. **Mode collapse**: Use minibatch discrimination, feature matching
2. **Training instability**: Try Wasserstein GAN, spectral normalization
3. **Vanishing gradients**: Use least squares GAN, relativistic GAN

### General Tips
- Start with small models and simple datasets (MNIST)
- Monitor both reconstruction and generation quality
- Use appropriate learning rates (often lower for GANs)
- Visualize samples frequently during training
- Be patient - generative models can take time to train

## Extensions and Further Reading

### Advanced GAN Architectures
- **DCGAN**: Deep Convolutional GAN (stable CNN-based GAN)
- **WGAN**: Wasserstein GAN (improved training stability)
- **StyleGAN**: High-resolution, style-controlled generation
- **Progressive GAN**: Gradually increase resolution during training

### Conditional Generation
- **Conditional VAE/GAN**: Generate specific classes
- **pix2pix**: Image-to-image translation with paired data
- **CycleGAN**: Unpaired image-to-image translation

### Modern Approaches
- **Diffusion Models**: Current state-of-the-art for image generation
- **VQ-VAE**: Discrete latent representations
- **Normalizing Flows**: Exact likelihood estimation

### Recommended Papers
1. Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
2. Goodfellow et al. (2014) - "Generative Adversarial Networks"
3. Radford et al. (2015) - "Unsupervised Representation Learning with DCGANs"
4. Arjovsky et al. (2017) - "Wasserstein GAN"
5. Karras et al. (2019) - "A Style-Based Generator Architecture for GANs"

### Tutorials and Resources
- PyTorch GAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- VAE Tutorial: https://arxiv.org/abs/1606.05908
- GAN Lab (Interactive): https://poloclub.github.io/ganlab/
- Papers with Code: https://paperswithcode.com/task/image-generation

## Assessment Suggestions

### Formative Assessment
- Complete workshop activities
- Answer reflection questions
- Generate samples from trained models
- Explain key concepts to peers

### Summative Assessment
- Implement a GAN variant (DCGAN, WGAN)
- Train generative model on new dataset
- Compare VAE and GAN quantitatively (FID, IS)
- Write report on generated samples quality
- Design application using generative models

### Project Ideas
1. **Face Generation**: Train GAN on celebrity faces dataset
2. **Style Transfer**: Implement neural style transfer
3. **Anomaly Detection**: Use autoencoder for defect detection
4. **Data Augmentation**: Generate synthetic training data
5. **Image Inpainting**: Fill in missing regions using generative model

## Building the Presentation

To rebuild the PDF from LaTeX source:

```bash
cd 11-GenerativeModels
pdflatex lecture11_generative_models.tex
pdflatex lecture11_generative_models.tex  # Run twice for references
```

## Directory Structure

```
11-GenerativeModels/
├── README.md                              # This file
├── lecture11_generative_models.tex        # LaTeX source
├── lecture11_generative_models.pdf        # Compiled presentation
├── workshop.ipynb                         # Interactive Jupyter notebook
├── requirements.txt                       # Python dependencies
├── figures/                               # All visualization assets
│   ├── core/                             # Core concept figures
│   │   ├── autoencoder_architecture.png
│   │   ├── vae_architecture.png
│   │   ├── gan_architecture.png
│   │   ├── latent_space_comparison.png
│   │   ├── vae_reparameterization.png
│   │   └── gan_training_dynamics.png
│   ├── advanced/                         # Advanced topic figures
│   │   ├── dcgan_architecture.png
│   │   ├── conditional_generation.png
│   │   ├── gan_mode_collapse.png
│   │   ├── vae_interpolation.png
│   │   └── latent_space_exploration.png
│   └── applications/                     # Application figures
│       ├── image_generation_comparison.png
│       ├── anomaly_detection.png
│       ├── data_augmentation.png
│       ├── image_editing.png
│       └── style_transfer.png
└── scripts/                              # Figure generation scripts
    ├── generate_core_figures.py
    ├── generate_advanced_figures.py
    ├── generate_application_figures.py
    └── generate_all_figures.py
```

## License

This educational material is provided for academic use in CMSC 178 - Introduction to Image Processing.

## Contact

For questions or issues with this lecture material, please contact the course instructor or teaching assistants.

## Acknowledgments

This lecture draws inspiration from:
- Stanford CS231n: Deep Learning for Computer Vision
- MIT 6.S191: Introduction to Deep Learning
- Original papers by Kingma, Goodfellow, and others
- PyTorch tutorials and documentation

---

**Last Updated**: October 2025
**Version**: 1.0
