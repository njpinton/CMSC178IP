#!/usr/bin/env python3
"""
Advanced Image Restoration and Geometric Processing Techniques
CMSC 178IP - Digital Image Processing

This script generates figures demonstrating advanced restoration methods,
motion deblurring, inpainting, and complex geometric transformations.

Author: Noel Jeffrey Pinton
Course: CMSC 178IP - Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage, signal
from skimage import data, restoration, transform, morphology, segmentation
from skimage.restoration import inpaint
from skimage.filters import unsharp_mask
import os

# Set up matplotlib for consistent styling
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme
PRIMARY_BLUE = '#144B8C'
SECONDARY_BLUE = '#4682B4'
ACCENT_ORANGE = '#E65F2D'

def ensure_output_dir():
    """Ensure the figures directory exists."""
    os.makedirs('../figures', exist_ok=True)

def motion_blur_restoration():
    """Demonstrate motion blur and restoration techniques."""
    # Load test image - use immunohistochemistry for detailed medical-like imagery
    image = data.immunohistochemistry()
    # Convert to grayscale for blur analysis
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create motion blur kernel
    def create_motion_blur_kernel(length, angle):
        """Create a motion blur kernel."""
        kernel = np.zeros((length, length))
        center = length // 2

        # Calculate line endpoints
        dx = int(center * np.cos(np.radians(angle)))
        dy = int(center * np.sin(np.radians(angle)))

        # Draw line in kernel
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)

        # Normalize
        kernel = kernel / np.sum(kernel)
        return kernel

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Motion Blur and Restoration', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Create motion blur
    motion_kernel = create_motion_blur_kernel(15, 45)
    blurred = signal.convolve2d(image, motion_kernel, mode='same', boundary='symm')

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Motion Blurred')
    axes[0, 1].axis('off')

    # Add noise to blurred image
    blurred_noisy = blurred + np.random.normal(0, 5, blurred.shape)

    axes[0, 2].imshow(blurred_noisy, cmap='gray')
    axes[0, 2].set_title('Blurred + Noise')
    axes[0, 2].axis('off')

    # Wiener deconvolution
    try:
        wiener_restored = restoration.wiener(blurred_noisy, motion_kernel, 0.01)
        axes[0, 3].imshow(wiener_restored, cmap='gray')
        axes[0, 3].set_title('Wiener Deconvolution')
        axes[0, 3].axis('off')
    except:
        # Fallback if wiener doesn't work
        axes[0, 3].imshow(blurred_noisy, cmap='gray')
        axes[0, 3].set_title('Restoration Failed')
        axes[0, 3].axis('off')

    # Richardson-Lucy deconvolution
    try:
        rl_restored = restoration.richardson_lucy(blurred_noisy, motion_kernel, iterations=30)
        axes[1, 0].imshow(rl_restored, cmap='gray')
        axes[1, 0].set_title('Richardson-Lucy')
        axes[1, 0].axis('off')
    except:
        axes[1, 0].imshow(blurred_noisy, cmap='gray')
        axes[1, 0].set_title('R-L Restoration')
        axes[1, 0].axis('off')

    # Unsharp masking
    unsharp_restored = unsharp_mask(blurred_noisy, radius=1, amount=2)
    axes[1, 1].imshow(unsharp_restored, cmap='gray')
    axes[1, 1].set_title('Unsharp Masking')
    axes[1, 1].axis('off')

    # Show the motion kernel
    axes[1, 2].imshow(motion_kernel, cmap='hot')
    axes[1, 2].set_title('Motion Blur Kernel')
    axes[1, 2].axis('off')

    # Frequency domain view
    kernel_fft = np.fft.fftshift(np.fft.fft2(motion_kernel, s=image.shape))
    axes[1, 3].imshow(np.log(1 + np.abs(kernel_fft)), cmap='viridis')
    axes[1, 3].set_title('Kernel Frequency Response')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/motion_blur_restoration.png', dpi=300, bbox_inches='tight')
    plt.close()

def image_inpainting():
    """Demonstrate image inpainting techniques."""
    # Load test image - use rocket for interesting geometric structures
    image = data.rocket()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a mask for inpainting (simulate damaged areas)
    mask = np.zeros(gray_image.shape, dtype=bool)

    # Add some rectangular damaged areas
    mask[100:150, 200:250] = True
    mask[300:320, 100:200] = True
    mask[250:300, 350:400] = True

    # Add some circular damaged areas
    y, x = np.ogrid[:gray_image.shape[0], :gray_image.shape[1]]
    circle1 = (x - 150)**2 + (y - 200)**2 < 30**2
    circle2 = (x - 350)**2 + (y - 150)**2 < 25**2
    mask[circle1] = True
    mask[circle2] = True

    # Create damaged image
    damaged_image = gray_image.copy()
    damaged_image[mask] = 0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Inpainting Techniques', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Damaged image
    axes[0, 1].imshow(damaged_image, cmap='gray')
    axes[0, 1].set_title('Damaged Image')
    axes[0, 1].axis('off')

    # Mask
    axes[0, 2].imshow(mask, cmap='Reds')
    axes[0, 2].set_title('Damage Mask')
    axes[0, 2].axis('off')

    # Biharmonic inpainting
    try:
        biharmonic_result = inpaint.inpaint_biharmonic(damaged_image, mask)
        axes[1, 0].imshow(biharmonic_result, cmap='gray')
        axes[1, 0].set_title('Biharmonic Inpainting')
        axes[1, 0].axis('off')
    except:
        axes[1, 0].imshow(damaged_image, cmap='gray')
        axes[1, 0].set_title('Biharmonic (Failed)')
        axes[1, 0].axis('off')

    # OpenCV inpainting methods
    mask_uint8 = mask.astype(np.uint8) * 255

    # Telea method
    telea_result = cv2.inpaint(damaged_image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_TELEA)
    axes[1, 1].imshow(telea_result, cmap='gray')
    axes[1, 1].set_title('Telea Inpainting')
    axes[1, 1].axis('off')

    # Navier-Stokes method
    ns_result = cv2.inpaint(damaged_image.astype(np.uint8), mask_uint8, 3, cv2.INPAINT_NS)
    axes[1, 2].imshow(ns_result, cmap='gray')
    axes[1, 2].set_title('Navier-Stokes Inpainting')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/image_inpainting.png', dpi=300, bbox_inches='tight')
    plt.close()

def advanced_geometric_transforms():
    """Demonstrate advanced geometric transformation techniques."""
    # Create a structured pattern with grid and shapes for clear visualization
    def create_advanced_pattern(size=200):
        """Create a pattern optimized for advanced transformation visualization."""
        pattern = np.ones((size, size)) * 0.8

        # Create a coordinate grid
        for i in range(0, size, 20):
            pattern[i:i+1, :] = 0.2  # Horizontal lines
            pattern[:, i:i+1] = 0.2  # Vertical lines

        # Add concentric circles for radial transformations
        y, x = np.ogrid[:size, :size]
        center = (size//2, size//2)
        for radius in [30, 60, 90]:
            circle_mask = np.abs(np.sqrt((x - center[1])**2 + (y - center[0])**2) - radius) < 1
            pattern[circle_mask] = 0

        # Add directional markers
        # Horizontal arrow
        pattern[size//2-2:size//2+3, size//4:3*size//4] = 0.4
        pattern[size//2-5:size//2+6, 3*size//4-5:3*size//4] = 0

        # Vertical arrow
        pattern[size//4:3*size//4, size//2-2:size//2+3] = 0.4
        pattern[3*size//4-5:3*size//4, size//2-5:size//2+6] = 0

        # Corner reference points
        corner_size = 8
        pattern[5:5+corner_size, 5:5+corner_size] = 0
        pattern[5:5+corner_size, -5-corner_size:-5] = 0
        pattern[-5-corner_size:-5, 5:5+corner_size] = 0
        pattern[-5-corner_size:-5, -5-corner_size:-5] = 0

        return pattern

    def add_transformation_guides(ax, transform_type, original_shape):
        """Add visual guides specific to each transformation type."""
        h, w = original_shape

        if transform_type == 'polar':
            # Show radial lines for polar transformation
            center = (h//2, w//2)
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x_end = center[1] + (w//3) * np.cos(angle)
                y_end = center[0] + (h//3) * np.sin(angle)
                ax.plot([center[1], x_end], [center[0], y_end],
                       color=ACCENT_ORANGE, linewidth=1, alpha=0.7)

        elif transform_type == 'swirl':
            # Show spiral pattern
            center = (h//2, w//2)
            theta = np.linspace(0, 4*np.pi, 100)
            r = np.linspace(0, min(h,w)//3, 100)
            x = center[1] + r * np.cos(theta)
            y = center[0] + r * np.sin(theta)
            ax.plot(x, y, color=PRIMARY_BLUE, linewidth=2, alpha=0.8)

        elif transform_type == 'barrel':
            # Show distortion grid
            for i in range(0, h, h//4):
                ax.axhline(y=i, color=SECONDARY_BLUE, linestyle=':', alpha=0.6)
            for i in range(0, w, w//4):
                ax.axvline(x=i, color=SECONDARY_BLUE, linestyle=':', alpha=0.6)

        elif transform_type == 'elastic':
            # Show deformation vectors
            step = 40
            for y in range(step, h-step, step):
                for x in range(step, w-step, step):
                    # Random small displacement vectors
                    dx = np.random.normal(0, 5)
                    dy = np.random.normal(0, 5)
                    ax.arrow(x, y, dx, dy, head_width=3, head_length=3,
                           fc=ACCENT_ORANGE, ec=ACCENT_ORANGE, alpha=0.7)

    test_image = create_advanced_pattern()
    np.random.seed(42)  # Set seed for reproducibility

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Advanced Geometric Transformations with Visual Guides', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original Structured Pattern')
    axes[0, 0].axis('off')
    # Add coordinate system
    h, w = test_image.shape
    axes[0, 0].axhline(y=h//2, color=ACCENT_ORANGE, linestyle='--', alpha=0.7, linewidth=1)
    axes[0, 0].axvline(x=w//2, color=ACCENT_ORANGE, linestyle='--', alpha=0.7, linewidth=1)

    # Polar transformation
    def cartesian_to_polar(image):
        """Convert cartesian to polar coordinates."""
        h, w = image.shape
        center = (h // 2, w // 2)

        # Create polar coordinate arrays
        max_radius = min(center)
        theta = np.linspace(0, 2*np.pi, w)
        radius = np.linspace(0, max_radius, h)

        # Convert to cartesian coordinates
        theta_grid, radius_grid = np.meshgrid(theta, radius)
        x = center[1] + radius_grid * np.cos(theta_grid)
        y = center[0] + radius_grid * np.sin(theta_grid)

        # Sample the image
        from scipy.interpolate import RegularGridInterpolator
        coords = np.stack([np.arange(h), np.arange(w)], axis=-1)
        interp = RegularGridInterpolator((np.arange(h), np.arange(w)), image,
                                       bounds_error=False, fill_value=0)

        polar_coords = np.stack([y.ravel(), x.ravel()], axis=-1)
        polar_image = interp(polar_coords).reshape(h, w)

        return polar_image

    try:
        polar_image = cartesian_to_polar(test_image)
        axes[0, 1].imshow(polar_image, cmap='gray')
        axes[0, 1].set_title('Polar Transform')
        axes[0, 1].axis('off')
        add_transformation_guides(axes[0, 1], 'polar', test_image.shape)
    except:
        axes[0, 1].imshow(test_image, cmap='gray')
        axes[0, 1].set_title('Polar (Failed)')
        axes[0, 1].axis('off')

    # Log-polar transformation
    try:
        # Simple log-polar approximation
        log_polar = np.log(1 + polar_image)
        axes[0, 2].imshow(log_polar, cmap='gray')
        axes[0, 2].set_title('Log-Polar Transform')
        axes[0, 2].axis('off')
        # Add logarithmic scale indicators
        axes[0, 2].axhline(y=log_polar.shape[0]//2, color=PRIMARY_BLUE,
                          linestyle=':', alpha=0.7, linewidth=1)
    except:
        axes[0, 2].imshow(test_image, cmap='gray')
        axes[0, 2].set_title('Log-Polar')
        axes[0, 2].axis('off')

    # Barrel distortion
    def barrel_distortion(image, k1=0.3):
        """Apply barrel distortion."""
        h, w = image.shape
        center_x, center_y = w / 2, h / 2

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Normalize coordinates
        x_norm = (x - center_x) / center_x
        y_norm = (y - center_y) / center_y

        # Calculate radius
        r = np.sqrt(x_norm**2 + y_norm**2)

        # Apply barrel distortion
        r_distorted = r * (1 + k1 * r**2)

        # Convert back to image coordinates
        x_distorted = x_norm * r_distorted / (r + 1e-8) * center_x + center_x
        y_distorted = y_norm * r_distorted / (r + 1e-8) * center_y + center_y

        # Clip coordinates
        x_distorted = np.clip(x_distorted, 0, w-1)
        y_distorted = np.clip(y_distorted, 0, h-1)

        # Interpolate
        from scipy.interpolate import griddata
        points = np.column_stack([y.ravel(), x.ravel()])
        values = image.ravel()
        xi = np.column_stack([y_distorted.ravel(), x_distorted.ravel()])

        distorted = griddata(points, values, xi, method='linear', fill_value=0)
        return distorted.reshape(h, w)

    try:
        barrel_distorted = barrel_distortion(test_image)
        axes[0, 3].imshow(barrel_distorted, cmap='gray')
        axes[0, 3].set_title('Barrel Distortion')
        axes[0, 3].axis('off')
        add_transformation_guides(axes[0, 3], 'barrel', test_image.shape)
    except:
        axes[0, 3].imshow(test_image, cmap='gray')
        axes[0, 3].set_title('Barrel Distortion')
        axes[0, 3].axis('off')

    # Piecewise affine transformation
    try:
        # Define source and destination points
        rows, cols = test_image.shape
        src = np.array([[0, 0], [0, rows], [cols, 0], [cols, rows],
                       [cols/2, 0], [0, rows/2], [cols, rows/2], [cols/2, rows]])
        dst = np.array([[10, 10], [10, rows-10], [cols-10, 10], [cols-10, rows-10],
                       [cols/2+20, 0], [0, rows/2-10], [cols, rows/2+10], [cols/2-20, rows]])

        tform = transform.PiecewiseAffineTransform()
        tform.estimate(src, dst)
        piecewise_affine = transform.warp(test_image, tform)

        axes[1, 0].imshow(piecewise_affine, cmap='gray')
        axes[1, 0].set_title('Piecewise Affine')
        axes[1, 0].axis('off')
        # Show control points and displacement vectors
        for i, (s, d) in enumerate(zip(src, dst)):
            axes[1, 0].plot(s[0], s[1], 'o', color=ACCENT_ORANGE, markersize=4)
            axes[1, 0].plot(d[0], d[1], 's', color=PRIMARY_BLUE, markersize=4)
            axes[1, 0].arrow(s[0], s[1], d[0]-s[0], d[1]-s[1],
                           head_width=5, head_length=5, fc=SECONDARY_BLUE,
                           ec=SECONDARY_BLUE, alpha=0.7, linewidth=1)
    except:
        axes[1, 0].imshow(test_image, cmap='gray')
        axes[1, 0].set_title('Piecewise Affine')
        axes[1, 0].axis('off')

    # Swirl transformation
    swirled = transform.swirl(test_image, rotation=0, strength=10, radius=80)
    axes[1, 1].imshow(swirled, cmap='gray')
    axes[1, 1].set_title('Swirl Transform')
    axes[1, 1].axis('off')
    add_transformation_guides(axes[1, 1], 'swirl', test_image.shape)

    # Elastic deformation
    def elastic_transform(image, alpha=100, sigma=10):
        """Apply elastic deformation."""
        random_state = np.random.RandomState(42)
        shape = image.shape

        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    try:
        elastic_deformed = elastic_transform(test_image)
        axes[1, 2].imshow(elastic_deformed, cmap='gray')
        axes[1, 2].set_title('Elastic Deformation')
        axes[1, 2].axis('off')
        add_transformation_guides(axes[1, 2], 'elastic', test_image.shape)
    except:
        axes[1, 2].imshow(test_image, cmap='gray')
        axes[1, 2].set_title('Elastic Deformation')
        axes[1, 2].axis('off')

    # Mesh grid visualization for perspective
    # Create a grid pattern
    grid = np.zeros((200, 200))
    for i in range(0, 200, 20):
        grid[i, :] = 1
        grid[:, i] = 1

    # Apply perspective transform to grid
    corners = np.array([[0, 0], [0, 199], [199, 0], [199, 199]])
    new_corners = np.array([[30, 20], [10, 180], [170, 30], [190, 170]])
    tform_perspective = transform.ProjectiveTransform()
    tform_perspective.estimate(corners, new_corners)
    grid_perspective = transform.warp(grid, tform_perspective, output_shape=(200, 200))

    axes[1, 3].imshow(grid_perspective, cmap='gray')
    axes[1, 3].set_title('Grid Perspective')
    axes[1, 3].axis('off')
    # Add corner markers to show perspective transformation
    for i, corner in enumerate(new_corners):
        axes[1, 3].plot(corner[0], corner[1], 'o', color=ACCENT_ORANGE, markersize=6)
        axes[1, 3].plot(corners[i][0], corners[i][1], 's', color=PRIMARY_BLUE, markersize=4, alpha=0.7)

    plt.tight_layout()
    plt.savefig('../figures/advanced_geometric_transforms.png', dpi=300, bbox_inches='tight')
    plt.close()

def frequency_domain_restoration():
    """Demonstrate frequency domain restoration techniques."""
    # Load test image - use coffee for rich textures and patterns
    image = data.coffee()
    # Convert to grayscale for frequency analysis
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create degradation (blur + noise)
    # Gaussian blur kernel
    from scipy.signal.windows import gaussian
    blur_kernel = np.outer(gaussian(15, 3), gaussian(15, 3))
    blur_kernel = blur_kernel / blur_kernel.sum()

    # Apply blur
    blurred = signal.convolve2d(image, blur_kernel, mode='same', boundary='symm')

    # Add noise
    noisy_blurred = blurred + np.random.normal(0, 10, blurred.shape)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Frequency Domain Restoration', fontsize=16, fontweight='bold')

    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Degraded
    axes[0, 1].imshow(noisy_blurred, cmap='gray')
    axes[0, 1].set_title('Blurred + Noise')
    axes[0, 1].axis('off')

    # Frequency domain analysis
    # FFT of original
    fft_original = np.fft.fftshift(np.fft.fft2(image))
    axes[0, 2].imshow(np.log(1 + np.abs(fft_original)), cmap='viridis')
    axes[0, 2].set_title('Original FFT')
    axes[0, 2].axis('off')

    # FFT of degraded
    fft_degraded = np.fft.fftshift(np.fft.fft2(noisy_blurred))
    axes[0, 3].imshow(np.log(1 + np.abs(fft_degraded)), cmap='viridis')
    axes[0, 3].set_title('Degraded FFT')
    axes[0, 3].axis('off')

    # Inverse filtering (idealized)
    # Create frequency domain blur kernel
    h, w = image.shape
    blur_kernel_freq = np.zeros((h, w))
    kh, kw = blur_kernel.shape
    blur_kernel_freq[:kh, :kw] = blur_kernel
    blur_kernel_freq = np.fft.fftshift(np.fft.fft2(blur_kernel_freq))

    # Inverse filter
    inverse_filter = 1.0 / (blur_kernel_freq + 1e-10)  # Add small value to avoid division by zero

    # Apply inverse filtering
    fft_restored_inverse = fft_degraded * inverse_filter
    restored_inverse = np.real(np.fft.ifft2(np.fft.ifftshift(fft_restored_inverse)))

    axes[1, 0].imshow(np.clip(restored_inverse, 0, 255), cmap='gray')
    axes[1, 0].set_title('Inverse Filter')
    axes[1, 0].axis('off')

    # Wiener filtering in frequency domain
    # Estimate noise-to-signal ratio
    noise_power = np.var(noisy_blurred - blurred)
    signal_power = np.var(image)
    nsr = noise_power / signal_power

    # Wiener filter
    H = blur_kernel_freq
    wiener_filter = np.conj(H) / (np.abs(H)**2 + nsr)

    fft_restored_wiener = fft_degraded * wiener_filter
    restored_wiener = np.real(np.fft.ifft2(np.fft.ifftshift(fft_restored_wiener)))

    axes[1, 1].imshow(np.clip(restored_wiener, 0, 255), cmap='gray')
    axes[1, 1].set_title('Wiener Filter')
    axes[1, 1].axis('off')

    # Constrained least squares filtering
    # Laplacian operator for regularization
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_freq = np.zeros((h, w))
    laplacian_freq[0:3, 0:3] = laplacian
    laplacian_freq = np.fft.fftshift(np.fft.fft2(laplacian_freq))

    # CLS filter
    gamma = 0.01  # Regularization parameter
    cls_filter = np.conj(H) / (np.abs(H)**2 + gamma * np.abs(laplacian_freq)**2)

    fft_restored_cls = fft_degraded * cls_filter
    restored_cls = np.real(np.fft.ifft2(np.fft.ifftshift(fft_restored_cls)))

    axes[1, 2].imshow(np.clip(restored_cls, 0, 255), cmap='gray')
    axes[1, 2].set_title('Constrained LS')
    axes[1, 2].axis('off')

    # Comparison of filter responses
    freq_response_data = {
        'Inverse': np.abs(inverse_filter),
        'Wiener': np.abs(wiener_filter),
        'CLS': np.abs(cls_filter)
    }

    # Plot cross-section through center
    center = h // 2
    axes[1, 3].plot(np.abs(inverse_filter[center, :]), label='Inverse', linewidth=2, color=PRIMARY_BLUE)
    axes[1, 3].plot(np.abs(wiener_filter[center, :]), label='Wiener', linewidth=2, color=SECONDARY_BLUE)
    axes[1, 3].plot(np.abs(cls_filter[center, :]), label='CLS', linewidth=2, color=ACCENT_ORANGE)
    axes[1, 3].set_title('Filter Responses')
    axes[1, 3].set_xlabel('Frequency')
    axes[1, 3].set_ylabel('Magnitude')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/frequency_domain_restoration.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all advanced technique figures."""
    print("Generating advanced image restoration and geometric processing figures...")

    ensure_output_dir()

    # Generate all figures
    motion_blur_restoration()
    print("✓ Generated motion blur restoration figure")

    image_inpainting()
    print("✓ Generated image inpainting figure")

    advanced_geometric_transforms()
    print("✓ Generated advanced geometric transforms figure")

    frequency_domain_restoration()
    print("✓ Generated frequency domain restoration figure")

    print("Advanced techniques figures generation complete!")

if __name__ == "__main__":
    main()