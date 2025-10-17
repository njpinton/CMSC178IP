#!/usr/bin/env python3
"""
Core Generative Models Visualizations
Covers: Basic concepts, Autoencoders, VAE fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import data
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
from PIL import ImageFont

# Set professional style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Professional color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


def create_generative_vs_discriminative():
    """
    Illustrate the difference between generative and discriminative models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Discriminative model
    ax = axes[0]
    np.random.seed(42)

    # Generate two classes
    class1_x = np.random.randn(100) + 2
    class1_y = np.random.randn(100) + 2
    class2_x = np.random.randn(100) - 2
    class2_y = np.random.randn(100) - 2

    ax.scatter(class1_x, class1_y, c=colors[0], s=50, alpha=0.6, label='Class 1', edgecolors='black')
    ax.scatter(class2_x, class2_y, c=colors[1], s=50, alpha=0.6, label='Class 2', edgecolors='black')

    # Draw decision boundary
    x_boundary = np.linspace(-5, 5, 100)
    y_boundary = x_boundary
    ax.plot(x_boundary, y_boundary, 'k--', linewidth=3, label='Decision Boundary')

    ax.set_xlabel('Feature 1', fontweight='bold')
    ax.set_ylabel('Feature 2', fontweight='bold')
    ax.set_title('Discriminative Model\nLearns P(y|x) - Classification Boundary',
                 fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Generative model
    ax = axes[1]

    # Show probability distributions for each class
    from matplotlib.patches import Ellipse

    ax.scatter(class1_x, class1_y, c=colors[0], s=50, alpha=0.6, label='Class 1', edgecolors='black')
    ax.scatter(class2_x, class2_y, c=colors[1], s=50, alpha=0.6, label='Class 2', edgecolors='black')

    # Draw probability distributions
    ell1 = Ellipse((2, 2), width=4, height=4, angle=0,
                   facecolor=colors[0], alpha=0.2, edgecolor=colors[0], linewidth=3)
    ell2 = Ellipse((-2, -2), width=4, height=4, angle=0,
                   facecolor=colors[1], alpha=0.2, edgecolor=colors[1], linewidth=3)
    ax.add_patch(ell1)
    ax.add_patch(ell2)

    ax.set_xlabel('Feature 1', fontweight='bold')
    ax.set_ylabel('Feature 2', fontweight='bold')
    ax.set_title('Generative Model\nLearns P(x|y) - Data Distribution',
                 fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    plt.suptitle('Discriminative vs Generative Models', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/generative_vs_discriminative.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated generative_vs_discriminative.png")


def create_autoencoder_architecture():
    """
    Create colorful and appealing autoencoder architecture diagram for beginners
    """
    fig, ax = plt.subplots(figsize=(18, 8))

    # Define vibrant colors
    input_color = '#FF6B6B'      # Coral red
    encoder_color = '#4ECDC4'    # Turquoise
    latent_color = '#FFE66D'     # Yellow
    decoder_color = '#95E1D3'    # Mint green
    output_color = '#A8E6CF'     # Light green

    # Define layers with fun 3D isometric style
    layers_data = [
        {'x': 0.05, 'y': 0.25, 'w': 0.08, 'h': 0.50, 'color': input_color,
         'label': 'Input\nImage\n28×28', 'icon': '🖼️'},

        {'x': 0.20, 'y': 0.30, 'w': 0.07, 'h': 0.40, 'color': encoder_color,
         'label': 'Encode\n512', 'icon': '⚡'},

        {'x': 0.33, 'y': 0.35, 'w': 0.06, 'h': 0.30, 'color': encoder_color,
         'label': 'Encode\n256', 'icon': '⚡'},

        {'x': 0.45, 'y': 0.38, 'w': 0.05, 'h': 0.24, 'color': latent_color,
         'label': 'Latent\n32', 'icon': '💎'},

        {'x': 0.57, 'y': 0.35, 'w': 0.06, 'h': 0.30, 'color': decoder_color,
         'label': 'Decode\n256', 'icon': '🔄'},

        {'x': 0.70, 'y': 0.30, 'w': 0.07, 'h': 0.40, 'color': decoder_color,
         'label': 'Decode\n512', 'icon': '🔄'},

        {'x': 0.85, 'y': 0.25, 'w': 0.08, 'h': 0.50, 'color': output_color,
         'label': 'Output\nImage\n28×28', 'icon': '✨'},
    ]

    # Draw each layer with 3D effect and rounded corners
    for i, layer in enumerate(layers_data):
        # Shadow
        shadow_offset = 0.008
        shadow = plt.Rectangle(
            (layer['x'] + shadow_offset, layer['y'] - shadow_offset),
            layer['w'], layer['h'],
            facecolor='gray', edgecolor='none', alpha=0.3, zorder=1
        )
        ax.add_patch(shadow)

        # Main rectangle with rounded corners
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['w'], layer['h'],
            boxstyle="round,pad=0.01",
            facecolor=layer['color'],
            edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)

        # Inner border for depth
        inner_box = FancyBboxPatch(
            (layer['x'] + 0.005, layer['y'] + 0.005),
            layer['w'] - 0.01, layer['h'] - 0.01,
            boxstyle="round,pad=0.008",
            facecolor='none',
            edgecolor='white', linewidth=2, alpha=0.5, zorder=3
        )
        ax.add_patch(inner_box)

        # Icon and label
        icon_y = layer['y'] + layer['h'] * 0.65
        text_y = layer['y'] + layer['h'] * 0.35

        ax.text(layer['x'] + layer['w']/2, icon_y, layer['icon'],
               ha='center', va='center', fontsize=28, zorder=4)

        ax.text(layer['x'] + layer['w']/2, text_y, layer['label'],
               ha='center', va='center', fontsize=11, weight='bold',
               color='#2c3e50', zorder=4)

    # Draw connecting arrows with gradient effect
    arrow_props = dict(arrowstyle='->', lw=4, color='#34495e',
                      connectionstyle="arc3,rad=0", alpha=0.7)

    for i in range(len(layers_data) - 1):
        start_x = layers_data[i]['x'] + layers_data[i]['w']
        start_y = layers_data[i]['y'] + layers_data[i]['h']/2
        end_x = layers_data[i+1]['x']
        end_y = layers_data[i+1]['y'] + layers_data[i+1]['h']/2

        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=arrow_props)

    # Add section labels with colorful badges
    def add_badge(x, y, text, bg_color, emoji):
        badge = FancyBboxPatch(
            (x - 0.06, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01",
            facecolor=bg_color, edgecolor='white', linewidth=3, zorder=5
        )
        ax.add_patch(badge)
        ax.text(x - 0.025, y, emoji, fontsize=20, ha='center', va='center', zorder=6)
        ax.text(x + 0.03, y, text, fontsize=13, weight='bold',
               color='white', ha='center', va='center', zorder=6)

    add_badge(0.265, 0.85, 'ENCODER', '#3498db', '📥')
    add_badge(0.50, 0.85, 'BOTTLENECK', '#f39c12', '💎')
    add_badge(0.735, 0.85, 'DECODER', '#2ecc71', '📤')

    # Add reconstruction arrow at bottom
    ax.annotate('', xy=(0.09, 0.12), xytext=(0.89, 0.12),
               arrowprops=dict(arrowstyle='<->', lw=3, color='#e74c3c'))

    # Add fun reconstruction loss text
    loss_text = '🎯 Goal: Make Output ≈ Input!\nLoss = ||Output - Input||²'
    ax.text(0.49, 0.08, loss_text, ha='center', fontsize=12,
           weight='bold', color='#e74c3c',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffe6e6',
                    edgecolor='#e74c3c', linewidth=2))

    ax.set_xlim(0, 0.98)
    ax.set_ylim(0, 0.95)
    ax.axis('off')
    ax.set_title('🤖 Autoencoder: Compress & Reconstruct! 🤖',
                fontsize=20, weight='bold', color='#2c3e50', pad=20)

    plt.tight_layout()
    plt.savefig('../figures/autoencoder_architecture.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated autoencoder_architecture.png")


def create_latent_space_visualization():
    """
    Visualize concept of latent space representation
    """
    fig = plt.figure(figsize=(16, 6))

    # Original high-dimensional space (simulated with 2D)
    ax1 = plt.subplot(131)
    np.random.seed(42)

    # Generate MNIST-like digit patterns
    n_samples = 50
    digits = []
    labels = []
    for digit in range(10):
        angle = (digit / 10) * 2 * np.pi
        x = 3 * np.cos(angle) + np.random.randn(5) * 0.3
        y = 3 * np.sin(angle) + np.random.randn(5) * 0.3
        for i in range(5):
            ax1.scatter(x[i], y[i], c=[colors[digit % len(colors)]],
                       s=200, alpha=0.7, edgecolors='black', linewidth=2)
            ax1.text(x[i], y[i], str(digit), ha='center', va='center',
                    fontsize=8, weight='bold', color='white')

    ax1.set_xlabel('Pixel Dimension 1', fontweight='bold')
    ax1.set_ylabel('Pixel Dimension 2', fontweight='bold')
    ax1.set_title('Input Space\n(High-Dimensional: 784D)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    # Encoder arrow
    ax2 = plt.subplot(132)
    ax2.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
                arrowprops=dict(arrowstyle='->', lw=10, color=colors[0]))
    ax2.text(0.5, 0.65, 'ENCODER', ha='center', fontsize=16, weight='bold')
    ax2.text(0.5, 0.35, 'Compress\nDimensionality', ha='center', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Latent space (compressed representation)
    ax3 = plt.subplot(133)

    # Show compressed latent space with meaningful structure
    for digit in range(10):
        angle = (digit / 10) * 2 * np.pi
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        ax3.scatter(x, y, c=[colors[digit % len(colors)]],
                   s=300, alpha=0.8, edgecolors='black', linewidth=3)
        ax3.text(x, y, str(digit), ha='center', va='center',
                fontsize=12, weight='bold', color='white')

        # Add arrows showing continuity
        next_digit = (digit + 1) % 10
        next_angle = (next_digit / 10) * 2 * np.pi
        next_x = 2 * np.cos(next_angle)
        next_y = 2 * np.sin(next_angle)
        ax3.annotate('', xy=(next_x*0.95, next_y*0.95), xytext=(x*0.95, y*0.95),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5,
                                  connectionstyle="arc3,rad=0.3"))

    ax3.set_xlabel('Latent Dimension 1', fontweight='bold')
    ax3.set_ylabel('Latent Dimension 2', fontweight='bold')
    ax3.set_title('Latent Space\n(Low-Dimensional: 2D)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)

    plt.suptitle('Latent Space: Learned Compressed Representation',
                fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/latent_space_concept.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated latent_space_concept.png")


def create_vae_architecture():
    """
    Create colorful and beginner-friendly VAE architecture diagram
    """
    fig, ax = plt.subplots(figsize=(20, 9))

    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

    # Vibrant colors
    input_color = '#FF6B6B'
    encoder_color = '#4ECDC4'
    mu_color = '#A8E6CF'
    sigma_color = '#FFD93D'
    sample_color = '#FF8C94'
    decoder_color = '#95E1D3'
    output_color = '#C7CEEA'

    # ENCODER SECTION
    encoder_layers = [
        {'x': 0.03, 'y': 0.30, 'w': 0.06, 'h': 0.40, 'color': input_color,
         'label': 'Input\n28×28', 'icon': '🖼️'},
        {'x': 0.13, 'y': 0.32, 'w': 0.055, 'h': 0.36, 'color': encoder_color,
         'label': 'Enc\n512', 'icon': '⚡'},
        {'x': 0.22, 'y': 0.34, 'w': 0.05, 'h': 0.32, 'color': encoder_color,
         'label': 'Enc\n256', 'icon': '⚡'},
    ]

    for layer in encoder_layers:
        # Shadow
        shadow = plt.Rectangle(
            (layer['x'] + 0.006, layer['y'] - 0.006),
            layer['w'], layer['h'],
            facecolor='gray', alpha=0.3, zorder=1
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['w'], layer['h'],
            boxstyle="round,pad=0.008",
            facecolor=layer['color'], edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)

        # Icon and text
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.65,
               layer['icon'], ha='center', fontsize=24, zorder=3)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.32,
               layer['label'], ha='center', fontsize=10, weight='bold', color='#2c3e50', zorder=3)

    # MU and SIGMA boxes
    mu_y = 0.52
    sigma_y = 0.28

    for y_pos, color, label, icon in [(mu_y, mu_color, 'μ\nmean', '📊'),
                                       (sigma_y, sigma_color, 'σ²\nvariance', '📈')]:
        box = FancyBboxPatch(
            (0.32, y_pos), 0.055, 0.14,
            boxstyle="round,pad=0.008",
            facecolor=color, edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)
        ax.text(0.3475, y_pos + 0.09, icon, ha='center', fontsize=20, zorder=3)
        ax.text(0.3475, y_pos + 0.04, label, ha='center', fontsize=9,
               weight='bold', color='#2c3e50', zorder=3)

    # SAMPLING CIRCLE (Reparameterization Trick)
    sample_x = 0.47
    sample_y = 0.50

    # Outer glow
    for r in [0.10, 0.09, 0.08]:
        circle = Circle((sample_x, sample_y), r, facecolor=sample_color,
                       alpha=0.2 - (0.10 - r), zorder=1)
        ax.add_patch(circle)

    # Main circle
    circle = Circle((sample_x, sample_y), 0.07, facecolor=sample_color,
                   edgecolor='white', linewidth=4, zorder=2)
    ax.add_patch(circle)

    ax.text(sample_x, sample_y + 0.03, '🎲', ha='center', fontsize=28, zorder=3)
    ax.text(sample_x, sample_y - 0.03, 'z', ha='center', fontsize=14,
           weight='bold', color='white', zorder=3)

    # Magic formula box
    formula_box = FancyBboxPatch(
        (0.39, 0.68), 0.16, 0.12,
        boxstyle="round,pad=0.01",
        facecolor='#FFF3CD', edgecolor='#F39C12', linewidth=3, zorder=2
    )
    ax.add_patch(formula_box)
    ax.text(0.47, 0.78, '✨ Magic Trick! ✨', ha='center', fontsize=10,
           weight='bold', color='#E67E22', zorder=3)
    ax.text(0.47, 0.72, 'z = μ + σ·ε', ha='center', fontsize=12,
           style='italic', weight='bold', color='#E74C3C', zorder=3)

    # DECODER SECTION
    decoder_layers = [
        {'x': 0.60, 'y': 0.34, 'w': 0.05, 'h': 0.32, 'color': decoder_color,
         'label': 'Dec\n256', 'icon': '🔄'},
        {'x': 0.69, 'y': 0.32, 'w': 0.055, 'h': 0.36, 'color': decoder_color,
         'label': 'Dec\n512', 'icon': '🔄'},
        {'x': 0.79, 'y': 0.30, 'w': 0.06, 'h': 0.40, 'color': output_color,
         'label': 'Output\n28×28', 'icon': '✨'},
    ]

    for layer in decoder_layers:
        # Shadow
        shadow = plt.Rectangle(
            (layer['x'] + 0.006, layer['y'] - 0.006),
            layer['w'], layer['h'],
            facecolor='gray', alpha=0.3, zorder=1
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['w'], layer['h'],
            boxstyle="round,pad=0.008",
            facecolor=layer['color'], edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)

        # Icon and text
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.65,
               layer['icon'], ha='center', fontsize=24, zorder=3)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.32,
               layer['label'], ha='center', fontsize=10, weight='bold', color='#2c3e50', zorder=3)

    # ARROWS
    arrow_style = dict(arrowstyle='->', lw=3, color='#34495e', alpha=0.7)

    # Encoder arrows
    ax.annotate('', xy=(0.13, 0.50), xytext=(0.095, 0.50), arrowprops=arrow_style)
    ax.annotate('', xy=(0.22, 0.50), xytext=(0.190, 0.50), arrowprops=arrow_style)

    # To mu and sigma
    ax.annotate('', xy=(0.32, 0.59), xytext=(0.275, 0.52), arrowprops=arrow_style)
    ax.annotate('', xy=(0.32, 0.35), xytext=(0.275, 0.48), arrowprops=arrow_style)

    # To sampling
    ax.annotate('', xy=(0.42, 0.56), xytext=(0.38, 0.60), arrowprops=arrow_style)
    ax.annotate('', xy=(0.42, 0.44), xytext=(0.38, 0.37), arrowprops=arrow_style)

    # To decoder
    ax.annotate('', xy=(0.60, 0.50), xytext=(0.54, 0.50), arrowprops=arrow_style)
    ax.annotate('', xy=(0.69, 0.50), xytext=(0.655, 0.50), arrowprops=arrow_style)
    ax.annotate('', xy=(0.79, 0.50), xytext=(0.750, 0.50), arrowprops=arrow_style)

    # LABELS WITH BADGES
    def add_badge(x, y, text, bg_color, emoji):
        badge = FancyBboxPatch(
            (x - 0.055, y - 0.035), 0.11, 0.07,
            boxstyle="round,pad=0.01",
            facecolor=bg_color, edgecolor='white', linewidth=3, zorder=5
        )
        ax.add_patch(badge)
        ax.text(x - 0.022, y, emoji, fontsize=18, ha='center', va='center', zorder=6)
        ax.text(x + 0.027, y, text, fontsize=12, weight='bold',
               color='white', ha='center', va='center', zorder=6)

    add_badge(0.165, 0.88, 'ENCODER', '#3498db', '📥')
    add_badge(0.47, 0.88, 'SAMPLE', '#e74c3c', '🎲')
    add_badge(0.71, 0.88, 'DECODER', '#2ecc71', '📤')

    # LOSS INFO
    loss_box = FancyBboxPatch(
        (0.25, 0.08), 0.40, 0.12,
        boxstyle="round,pad=0.015",
        facecolor='#FFE6E6', edgecolor='#E74C3C', linewidth=3, zorder=2
    )
    ax.add_patch(loss_box)
    ax.text(0.45, 0.16, '🎯 Loss = Reconstruction + KL Divergence', ha='center',
           fontsize=13, weight='bold', color='#E74C3C', zorder=3)
    ax.text(0.45, 0.10, '(Make it look real + Keep z nice & smooth!)', ha='center',
           fontsize=10, style='italic', color='#C0392B', zorder=3)

    ax.set_xlim(0, 0.90)
    ax.set_ylim(0, 0.95)
    ax.axis('off')
    ax.set_title('🌟 VAE: Variational Autoencoder - Adding Randomness! 🌟',
                fontsize=20, weight='bold', color='#2c3e50', pad=20)

    plt.tight_layout()
    plt.savefig('../figures/vae_architecture.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated vae_architecture.png")


def create_vae_sampling_process():
    """
    Illustrate VAE sampling and generation process
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    np.random.seed(42)

    # Row 1: Encoding process
    # Original image
    ax = axes[0, 0]
    original = np.random.rand(28, 28)
    ax.imshow(original, cmap='gray')
    ax.set_title('1. Input Image\nx', fontweight='bold', fontsize=12)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor=colors[0], linewidth=4))

    # Encoding to distribution
    ax = axes[0, 1]
    x = np.linspace(-3, 3, 100)

    # Mean and variance from encoder
    mu = 0.5
    sigma = 0.8

    # Plot distribution
    from scipy.stats import norm
    y = norm.pdf(x, mu, sigma)
    ax.fill_between(x, y, alpha=0.3, color=colors[3])
    ax.plot(x, y, linewidth=3, color=colors[3])
    ax.axvline(mu, color=colors[1], linestyle='--', linewidth=2, label=f'μ = {mu}')
    ax.axvline(mu-sigma, color=colors[4], linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(mu+sigma, color=colors[4], linestyle=':', linewidth=2, alpha=0.7, label=f'σ = {sigma}')

    ax.set_title('2. Encode to\nDistribution q(z|x)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Latent Variable z', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample from distribution
    ax = axes[0, 2]
    samples = np.random.normal(mu, sigma, 1000)
    ax.hist(samples, bins=30, alpha=0.7, color=colors[3], edgecolor='black')

    # Highlight one sample
    sample_z = mu + sigma * np.random.randn()
    ax.axvline(sample_z, color=colors[1], linestyle='--', linewidth=3, label=f'z = {sample_z:.2f}')

    ax.set_title('3. Sample Latent\nCode z ~ q(z|x)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Latent Variable z', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Decoding process
    # Decode sample
    ax = axes[1, 0]
    reconstructed = ndimage.gaussian_filter(original, sigma=1.0)
    ax.imshow(reconstructed, cmap='gray')
    ax.set_title('4. Decode to\nReconstruction x̂', fontweight='bold', fontsize=12)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor=colors[2], linewidth=4))

    # Generate new sample from prior
    ax = axes[1, 1]
    x_prior = np.linspace(-3, 3, 100)
    y_prior = norm.pdf(x_prior, 0, 1)  # Prior N(0,1)
    ax.fill_between(x_prior, y_prior, alpha=0.3, color=colors[5])
    ax.plot(x_prior, y_prior, linewidth=3, color=colors[5])

    # Sample from prior
    new_sample = np.random.randn()
    ax.axvline(new_sample, color=colors[1], linestyle='--', linewidth=3,
              label=f'z* = {new_sample:.2f}')

    ax.set_title('5. Sample from\nPrior p(z) ~ N(0,1)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Latent Variable z', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Generate new image
    ax = axes[1, 2]
    generated = np.random.rand(28, 28)
    generated = ndimage.gaussian_filter(generated, sigma=1.5)
    ax.imshow(generated, cmap='gray')
    ax.set_title('6. Generate New\nImage x_new', fontweight='bold', fontsize=12)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor=colors[5], linewidth=4))

    plt.suptitle('VAE: Encoding (Inference) and Generation Process',
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/vae_sampling_process.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated vae_sampling_process.png")


def main():
    """Generate all core concept figures"""
    print("\n" + "="*70)
    print("Generating Core Generative Models Figures")
    print("="*70)

    create_generative_vs_discriminative()
    create_autoencoder_architecture()
    create_latent_space_visualization()
    create_vae_architecture()
    create_vae_sampling_process()

    print("\n" + "="*70)
    print("✓ Core figures generated successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
