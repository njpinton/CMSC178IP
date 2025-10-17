#!/usr/bin/env python3
"""
Generative Models Applications
Covers: Real-world examples, MNIST generation, image-to-image translation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import data, transform, io
import warnings
import requests
from io import BytesIO
warnings.filterwarnings('ignore')

# Set professional style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Professional color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


def fetch_picsum_image(image_id=None, width=300, height=300, grayscale=False):
    """
    Fetch an image from Picsum (Lorem Picsum - placeholder images)

    Args:
        image_id: Specific image ID (optional, random if None)
        width: Image width
        height: Image height
        grayscale: Whether to fetch grayscale version

    Returns:
        numpy array of the image
    """
    try:
        if image_id is not None:
            url = f"https://picsum.photos/id/{image_id}/{width}/{height}"
        else:
            url = f"https://picsum.photos/{width}/{height}"

        if grayscale:
            url += "?grayscale"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = io.imread(BytesIO(response.content))

        # Convert to float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0

        # If grayscale requested but got RGB, convert
        if grayscale and len(img.shape) == 3:
            img = np.mean(img, axis=2)

        return img
    except Exception as e:
        print(f"Warning: Could not fetch from Picsum ({e}), using fallback image")
        # Fallback to skimage data
        fallback = data.camera()
        fallback = transform.resize(fallback, (height, width))
        return fallback / 255.0 if fallback.max() > 1 else fallback


def create_generation_examples():
    """
    Show examples of generated vs real images (simulated MNIST-like)
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    np.random.seed(42)

    # Top row: Real images (simulated MNIST-like digits)
    real_digits = [
        create_digit(0), create_digit(1), create_digit(2),
        create_digit(3), create_digit(4)
    ]

    for i, digit_img in enumerate(real_digits):
        axes[0, i].imshow(digit_img, cmap='gray')
        axes[0, i].set_title(f'Real\nDigit {i}', fontsize=11, weight='bold', color=colors[0])
        axes[0, i].axis('off')
        # Green border for real
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor(colors[0])
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Bottom row: Generated images (slightly modified/blurred)
    for i, digit_img in enumerate(real_digits):
        # Add some noise and blur to simulate generated
        generated = digit_img + np.random.randn(28, 28) * 0.1
        generated = ndimage.gaussian_filter(generated, sigma=0.5)
        generated = np.clip(generated, 0, 1)

        axes[1, i].imshow(generated, cmap='gray')
        axes[1, i].set_title(f'Generated\nDigit {i}', fontsize=11, weight='bold', color=colors[2])
        axes[1, i].axis('off')
        # Orange border for generated
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor(colors[2])
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.suptitle('Real vs Generated Images (MNIST-style)', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/generation_examples.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated generation_examples.png")


def create_digit(digit):
    """Helper function to create simple digit-like images"""
    img = np.zeros((28, 28))

    if digit == 0:
        # Circle
        y, x = np.ogrid[-14:14, -14:14]
        mask = (x**2 + y**2 <= 100) & (x**2 + y**2 >= 64)
        img[mask] = 1.0
    elif digit == 1:
        # Vertical line
        img[5:23, 12:16] = 1.0
    elif digit == 2:
        # S-shape approximation
        img[6:10, 8:20] = 1.0
        img[10:14, 16:20] = 1.0
        img[14:18, 8:16] = 1.0
        img[18:22, 8:12] = 1.0
    elif digit == 3:
        # E-shape approximation
        img[6:10, 10:20] = 1.0
        img[12:15, 10:18] = 1.0
        img[18:22, 10:20] = 1.0
        img[6:22, 18:20] = 1.0
    elif digit == 4:
        # 4-shape
        img[6:15, 8:11] = 1.0
        img[13:16, 8:20] = 1.0
        img[6:22, 17:20] = 1.0
    else:
        # Random pattern
        img[8:20, 10:18] = 1.0

    # Add some randomness
    img = img + np.random.randn(28, 28) * 0.05
    img = np.clip(img, 0, 1)
    img = ndimage.gaussian_filter(img, sigma=0.8)

    return img


def create_latent_space_interpolation():
    """
    Show interpolation in latent space
    """
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))

    np.random.seed(42)

    # Create two endpoint digits
    digit_a = create_digit(0)  # Circle
    digit_b = create_digit(1)  # Line

    # Interpolate in "latent space" (simple linear interpolation for visualization)
    alphas = np.linspace(0, 1, 6)

    for i, alpha in enumerate(alphas):
        # Interpolate
        interpolated = (1 - alpha) * digit_a + alpha * digit_b
        interpolated = ndimage.gaussian_filter(interpolated, sigma=0.5)

        axes[0, i].imshow(interpolated, cmap='gray')
        axes[0, i].set_title(f'α={alpha:.2f}', fontsize=11, weight='bold')
        axes[0, i].axis('off')

    # Another interpolation pair
    digit_c = create_digit(2)
    digit_d = create_digit(3)

    for i, alpha in enumerate(alphas):
        interpolated = (1 - alpha) * digit_c + alpha * digit_d
        interpolated = ndimage.gaussian_filter(interpolated, sigma=0.5)

        axes[1, i].imshow(interpolated, cmap='gray')
        axes[1, i].set_title(f'α={alpha:.2f}', fontsize=11, weight='bold')
        axes[1, i].axis('off')

    # Add labels
    fig.text(0.02, 0.75, 'Interpolation 1:\n0 → 1', ha='center', va='center',
            fontsize=12, weight='bold', rotation=90)
    fig.text(0.02, 0.25, 'Interpolation 2:\n2 → 3', ha='center', va='center',
            fontsize=12, weight='bold', rotation=90)

    plt.suptitle('Latent Space Interpolation - Smooth Transitions', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig('../figures/latent_interpolation.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated latent_interpolation.png")


def create_conditional_generation():
    """
    Illustrate conditional generation (class-conditional)
    """
    fig, axes = plt.subplots(3, 5, figsize=(14, 9))

    np.random.seed(42)

    # Generate conditioned on class labels
    for row in range(3):
        for col in range(5):
            digit_class = (row * 5 + col) % 10

            # Generate digit with some variation
            base_digit = create_digit(digit_class % 5)
            # Add variation
            variation = np.random.randn(28, 28) * 0.08
            generated = base_digit + variation
            generated = ndimage.gaussian_filter(generated, sigma=0.6)
            generated = np.clip(generated, 0, 1)

            axes[row, col].imshow(generated, cmap='gray')
            axes[row, col].set_title(f'Class {digit_class}', fontsize=11, weight='bold',
                                    color=colors[digit_class % len(colors)])
            axes[row, col].axis('off')

            # Colored border by class
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(colors[digit_class % len(colors)])
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.suptitle('Conditional Generation - Control Output Class', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/conditional_generation.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated conditional_generation.png")


def create_style_transfer_concept():
    """
    Illustrate style transfer concept using real images from Picsum
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Load real images from Picsum
    # Using specific IDs for reproducibility - landscape and architecture
    content_img = fetch_picsum_image(image_id=1015, width=256, height=256, grayscale=True)
    style_img1 = fetch_picsum_image(image_id=1018, width=256, height=256, grayscale=True)

    # Create "style" images with different filters applied to style images
    style1 = ndimage.gaussian_filter(style_img1, sigma=3)
    style2 = ndimage.sobel(content_img)

    # Content image
    axes[0, 0].imshow(content_img, cmap='gray')
    axes[0, 0].set_title('Content Image', fontsize=14, weight='bold', color=colors[0])
    axes[0, 0].axis('off')

    # Style 1
    axes[0, 1].imshow(style1, cmap='hot')
    axes[0, 1].set_title('Style A\n(Smooth)', fontsize=14, weight='bold', color=colors[3])
    axes[0, 1].axis('off')

    # Result 1
    result1 = 0.7 * content_img + 0.3 * style1
    axes[0, 2].imshow(result1, cmap='gray')
    axes[0, 2].set_title('Generated\n(Content + Style A)', fontsize=14, weight='bold', color=colors[2])
    axes[0, 2].axis('off')

    # Content image again
    axes[1, 0].imshow(content_img, cmap='gray')
    axes[1, 0].set_title('Content Image', fontsize=14, weight='bold', color=colors[0])
    axes[1, 0].axis('off')

    # Style 2
    axes[1, 1].imshow(np.abs(style2), cmap='copper')
    axes[1, 1].set_title('Style B\n(Edges)', fontsize=14, weight='bold', color=colors[3])
    axes[1, 1].axis('off')

    # Result 2
    result2 = content_img + 0.3 * np.abs(style2)
    axes[1, 2].imshow(result2, cmap='gray')
    axes[1, 2].set_title('Generated\n(Content + Style B)', fontsize=14, weight='bold', color=colors[2])
    axes[1, 2].axis('off')

    # Add arrows
    for row in range(2):
        # Content to result arrow
        axes[row, 0].annotate('', xy=(1.1, 0.5), xytext=(0.9, 0.5),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', lw=3, color=colors[0]))
        # Style to result arrow
        axes[row, 1].annotate('', xy=(1.1, 0.5), xytext=(0.9, 0.5),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', lw=3, color=colors[3]))

    plt.suptitle('Style Transfer Concept (Simplified)', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/style_transfer_concept.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated style_transfer_concept.png")


def create_image_to_image_translation():
    """
    Demonstrate image-to-image translation concept using Picsum images
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    np.random.seed(42)

    # Fetch diverse images from Picsum
    image_ids = [1025, 1043, 1047, 1050]  # Different scene types

    for idx, img_id in enumerate(image_ids):
        # Original image (color)
        img_color = fetch_picsum_image(image_id=img_id, width=200, height=200, grayscale=False)

        # Translation 1: Grayscale (simulating domain transfer)
        if len(img_color.shape) == 3:
            img_gray = np.mean(img_color, axis=2)
        else:
            img_gray = img_color

        # Original
        if len(img_color.shape) == 3:
            axes[0, idx].imshow(img_color)
        else:
            axes[0, idx].imshow(img_color, cmap='gray')
        axes[0, idx].set_title(f'Domain A\n(Color)', fontsize=11, weight='bold', color=colors[0])
        axes[0, idx].axis('off')

        # Border
        for spine in axes[0, idx].spines.values():
            spine.set_edgecolor(colors[0])
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Translated
        axes[1, idx].imshow(img_gray, cmap='gray')
        axes[1, idx].set_title(f'Domain B\n(Grayscale)', fontsize=11, weight='bold', color=colors[1])
        axes[1, idx].axis('off')

        # Border
        for spine in axes[1, idx].spines.values():
            spine.set_edgecolor(colors[1])
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Add arrows
    fig.text(0.5, 0.5, '⬇ Translation ⬇', ha='center', va='center',
            fontsize=16, weight='bold', color=colors[2],
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor=colors[2], linewidth=3))

    plt.suptitle('Image-to-Image Translation (Pix2Pix / CycleGAN)', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/image_to_image_translation.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated image_to_image_translation.png")


def create_applications_overview():
    """
    Overview of generative model applications
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    applications = [
        {'title': 'Image Generation', 'desc': '• Generate realistic images\n• MNIST, CIFAR, faces\n• Art creation', 'icon': '🖼️'},
        {'title': 'Image-to-Image', 'desc': '• Style transfer\n• Colorization\n• Super-resolution', 'icon': '🎨'},
        {'title': 'Data Augmentation', 'desc': '• Synthetic training data\n• Rare class generation\n• Domain adaptation', 'icon': '📊'},
        {'title': 'Anomaly Detection', 'desc': '• Reconstruction error\n• Out-of-distribution\n• Quality control', 'icon': '🔍'},
        {'title': 'Text-to-Image', 'desc': '• Generate from caption\n• Conditional synthesis\n• Creative tools', 'icon': '💬'},
        {'title': 'Medical Imaging', 'desc': '• Synthetic MRI/CT\n• Data privacy\n• Rare disease simulation', 'icon': '🏥'},
    ]

    for idx, (ax, app) in enumerate(zip(axes.flat, applications)):
        # Background color
        bg_color = colors[idx % len(colors)]

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg_color, alpha=0.2))

        # Icon
        ax.text(0.5, 0.75, app['icon'], ha='center', va='center', fontsize=60)

        # Title
        ax.text(0.5, 0.50, app['title'], ha='center', va='center',
               fontsize=14, weight='bold', color='black')

        # Description
        ax.text(0.5, 0.20, app['desc'], ha='center', va='center',
               fontsize=10, color='black',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Generative Models: Real-World Applications', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/applications_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated applications_overview.png")


def create_training_tips():
    """
    Create visualization of training tips and best practices
    """
    fig = plt.figure(figsize=(14, 10))

    # VAE Tips
    ax1 = plt.subplot(2, 2, 1)
    vae_tips = [
        '✓ Use β-VAE for better disentanglement',
        '✓ Balance reconstruction vs KL term',
        '✓ Warm-up KL weight gradually',
        '✓ Monitor ELBO during training',
        '✓ Use appropriate prior (N(0,1))',
    ]

    y_pos = 0.9
    ax1.text(0.5, y_pos, 'VAE Training Tips', ha='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[3], alpha=0.3))

    y_pos -= 0.15
    for tip in vae_tips:
        ax1.text(0.1, y_pos, tip, ha='left', fontsize=11, verticalalignment='top')
        y_pos -= 0.15

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # GAN Tips
    ax2 = plt.subplot(2, 2, 2)
    gan_tips = [
        '✓ Balance D and G training',
        '✓ Use label smoothing',
        '✓ Add noise to discriminator inputs',
        '✓ Use batch normalization',
        '✓ Monitor mode collapse',
    ]

    y_pos = 0.9
    ax2.text(0.5, y_pos, 'GAN Training Tips', ha='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[1], alpha=0.3))

    y_pos -= 0.15
    for tip in gan_tips:
        ax2.text(0.1, y_pos, tip, ha='left', fontsize=11, verticalalignment='top')
        y_pos -= 0.15

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Common Pitfalls
    ax3 = plt.subplot(2, 2, 3)
    pitfalls = [
        '✗ Insufficient model capacity',
        '✗ Poor initialization',
        '✗ Learning rate too high/low',
        '✗ Not monitoring metrics',
        '✗ Overfitting to training set',
    ]

    y_pos = 0.9
    ax3.text(0.5, y_pos, 'Common Pitfalls', ha='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[1], alpha=0.5))

    y_pos -= 0.15
    for pitfall in pitfalls:
        ax3.text(0.1, y_pos, pitfall, ha='left', fontsize=11, verticalalignment='top', color='darkred')
        y_pos -= 0.15

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Evaluation Metrics
    ax4 = plt.subplot(2, 2, 4)
    metrics = [
        '📊 Inception Score (IS)',
        '📊 Fréchet Inception Distance (FID)',
        '📊 Reconstruction Error',
        '📊 Visual Quality Assessment',
        '📊 Diversity Metrics',
    ]

    y_pos = 0.9
    ax4.text(0.5, y_pos, 'Evaluation Metrics', ha='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[2], alpha=0.3))

    y_pos -= 0.15
    for metric in metrics:
        ax4.text(0.1, y_pos, metric, ha='left', fontsize=11, verticalalignment='top')
        y_pos -= 0.15

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.suptitle('Training Tips and Best Practices', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../figures/training_tips.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated training_tips.png")


def main():
    """Generate all application figures"""
    print("\n" + "="*70)
    print("Generating Generative Models Applications Figures")
    print("="*70)

    create_generation_examples()
    create_latent_space_interpolation()
    create_conditional_generation()
    create_style_transfer_concept()
    create_image_to_image_translation()
    create_applications_overview()
    create_training_tips()

    print("\n" + "="*70)
    print("✓ Application figures generated successfully!")
    print("  Note: Some figures use images from Picsum.photos")
    print("="*70)


if __name__ == '__main__':
    main()
