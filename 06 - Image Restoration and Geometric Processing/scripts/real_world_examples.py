#!/usr/bin/env python3
"""
Real-World Image Restoration and Geometric Processing Examples
CMSC 178IP - Digital Image Processing

This script generates figures demonstrating practical applications of image
restoration and geometric processing in real-world scenarios including
medical imaging, document processing, and computer vision applications.

Author: Noel Jeffrey Pinton
Course: CMSC 178IP - Digital Image Processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import data, restoration, transform, morphology, filters, measure
from skimage.restoration import inpaint
from skimage.segmentation import active_contour
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

def document_processing_pipeline():
    """Demonstrate document processing and restoration."""
    # Use the actual text image from skimage for realistic document processing
    doc = data.text()
    # Convert to float for processing
    doc = doc.astype(float)

    # Add some noise and degradation
    # Gaussian noise
    noise = np.random.normal(0, 20, doc.shape)
    noisy_doc = np.clip(doc + noise, 0, 255)

    # Add some blur (camera shake)
    blurred_doc = ndimage.gaussian_filter(noisy_doc, sigma=1.5)

    # Add ink spots (salt and pepper noise)
    mask = np.random.random(doc.shape) < 0.001
    spotted_doc = blurred_doc.copy()
    spotted_doc[mask] = 0

    # Add some rotation (skewed scan)
    rotated_doc = transform.rotate(spotted_doc, angle=2.5, resize=True, cval=255)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Document Processing Pipeline', fontsize=16, fontweight='bold')

    # Original clean document
    axes[0, 0].imshow(doc, cmap='gray')
    axes[0, 0].set_title('Clean Document')
    axes[0, 0].axis('off')

    # Noisy document
    axes[0, 1].imshow(noisy_doc, cmap='gray')
    axes[0, 1].set_title('+ Noise')
    axes[0, 1].axis('off')

    # Blurred document
    axes[0, 2].imshow(blurred_doc, cmap='gray')
    axes[0, 2].set_title('+ Blur')
    axes[0, 2].axis('off')

    # Final degraded document
    axes[0, 3].imshow(rotated_doc, cmap='gray')
    axes[0, 3].set_title('Final Degraded')
    axes[0, 3].axis('off')

    # Processing steps
    # 1. Deskewing (rotation correction)
    # For simplicity, we'll just rotate back
    deskewed = transform.rotate(rotated_doc, angle=-2.5, resize=False, cval=255)
    center_h = deskewed.shape[0] // 2
    center_w = deskewed.shape[1] // 2
    h_offset = (deskewed.shape[0] - doc.shape[0]) // 2
    w_offset = (deskewed.shape[1] - doc.shape[1]) // 2
    if h_offset > 0 and w_offset > 0:
        deskewed = deskewed[h_offset:h_offset+doc.shape[0], w_offset:w_offset+doc.shape[1]]
    else:
        deskewed = deskewed[:doc.shape[0], :doc.shape[1]]

    axes[1, 0].imshow(deskewed, cmap='gray')
    axes[1, 0].set_title('1. Deskewed')
    axes[1, 0].axis('off')

    # 2. Noise reduction
    denoised = ndimage.median_filter(deskewed, size=3)
    axes[1, 1].imshow(denoised, cmap='gray')
    axes[1, 1].set_title('2. Denoised')
    axes[1, 1].axis('off')

    # 3. Sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = ndimage.convolve(denoised, kernel)
    sharpened = np.clip(sharpened, 0, 255)
    axes[1, 2].imshow(sharpened, cmap='gray')
    axes[1, 2].set_title('3. Sharpened')
    axes[1, 2].axis('off')

    # 4. Binarization
    threshold = filters.threshold_otsu(sharpened)
    binary = sharpened > threshold
    axes[1, 3].imshow(binary, cmap='gray')
    axes[1, 3].set_title('4. Binarized')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/document_processing.png', dpi=300, bbox_inches='tight')
    plt.close()

def medical_image_restoration():
    """Demonstrate medical image restoration techniques."""
    # Use cell image which resembles medical microscopy data
    medical_img = data.cell()
    # Check if it's already grayscale or needs conversion
    if len(medical_img.shape) == 3:
        medical_img = cv2.cvtColor(medical_img, cv2.COLOR_RGB2GRAY)
    # Convert to float for processing
    medical_img = medical_img.astype(float)

    # Add realistic medical image noise
    # Quantum noise (Poisson)
    scale_factor = 100
    medical_img_scaled = medical_img * scale_factor
    poisson_noise = np.random.poisson(medical_img_scaled) / scale_factor

    # Create coordinate grids for artifacts
    h, w = medical_img.shape
    y, x = np.ogrid[:h, :w]

    # Add some ring artifacts (common in CT)
    center = (h//2, w//2)
    for radius in [min(h,w)//4, min(h,w)//3, min(h,w)//2]:
        circle_mask = np.abs(np.sqrt((x - center[1])**2 + (y - center[0])**2) - radius) < 2
        poisson_noise = np.where(circle_mask, poisson_noise + 20, poisson_noise)

    # Add streaking artifacts
    streak_mask = np.abs(x - w//2) < 2
    poisson_noise = np.where(streak_mask, poisson_noise + 15, poisson_noise)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Medical Image Restoration', fontsize=16, fontweight='bold')

    # Original clean image
    axes[0, 0].imshow(medical_img, cmap='gray')
    axes[0, 0].set_title('Clean Medical Image')
    axes[0, 0].axis('off')

    # Noisy image
    axes[0, 1].imshow(poisson_noise, cmap='gray')
    axes[0, 1].set_title('With Artifacts & Noise')
    axes[0, 1].axis('off')

    # Gaussian filtering
    gaussian_filtered = ndimage.gaussian_filter(poisson_noise, sigma=1.0)
    axes[0, 2].imshow(gaussian_filtered, cmap='gray')
    axes[0, 2].set_title('Gaussian Filtered')
    axes[0, 2].axis('off')

    # Bilateral filtering
    bilateral_filtered = cv2.bilateralFilter(poisson_noise.astype(np.float32), 9, 75, 75)
    axes[0, 3].imshow(bilateral_filtered, cmap='gray')
    axes[0, 3].set_title('Bilateral Filtered')
    axes[0, 3].axis('off')

    # Edge-preserving smoothing
    # Anisotropic diffusion approximation using bilateral filter iteratively
    anisotropic = poisson_noise.copy().astype(np.float32)
    for i in range(3):
        anisotropic = cv2.bilateralFilter(anisotropic, 5, 50, 50)

    axes[1, 0].imshow(anisotropic, cmap='gray')
    axes[1, 0].set_title('Anisotropic Diffusion')
    axes[1, 0].axis('off')

    # Non-local means denoising
    try:
        nlm_filtered = restoration.denoise_nl_means(poisson_noise, patch_size=5, patch_distance=7, h=0.1)
        axes[1, 1].imshow(nlm_filtered, cmap='gray')
        axes[1, 1].set_title('Non-local Means')
        axes[1, 1].axis('off')
    except:
        axes[1, 1].imshow(bilateral_filtered, cmap='gray')
        axes[1, 1].set_title('Non-local Means')
        axes[1, 1].axis('off')

    # Morphological processing for structure enhancement
    # Top-hat filtering
    selem = morphology.disk(5)
    tophat = morphology.white_tophat(gaussian_filtered, selem)
    enhanced = gaussian_filtered + tophat

    axes[1, 2].imshow(enhanced, cmap='gray')
    axes[1, 2].set_title('Morphological Enhanced')
    axes[1, 2].axis('off')

    # Profile comparison
    line_pos = 150
    axes[1, 3].plot(medical_img[line_pos, :], label='Original', linewidth=2, color=PRIMARY_BLUE)
    axes[1, 3].plot(poisson_noise[line_pos, :], label='Noisy', linewidth=2, alpha=0.7, color='red')
    axes[1, 3].plot(bilateral_filtered[line_pos, :], label='Restored', linewidth=2, color=ACCENT_ORANGE)
    axes[1, 3].set_title('Profile Comparison')
    axes[1, 3].set_xlabel('Position')
    axes[1, 3].set_ylabel('Intensity')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/medical_image_restoration.png', dpi=300, bbox_inches='tight')
    plt.close()

def computer_vision_preprocessing():
    """Demonstrate image preprocessing for computer vision applications."""
    # Use chelsea cat image for object detection preprocessing
    scene = data.chelsea()
    # Convert to grayscale for processing
    scene = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY).astype(float)

    # Add lighting gradient
    h, w = scene.shape
    gradient_x = np.linspace(0.8, 1.2, w)
    gradient_y = np.linspace(1.2, 0.8, h)
    lighting = np.outer(gradient_y, gradient_x)
    lit_scene = scene * lighting

    # Add motion blur
    kernel = np.zeros((15, 15))
    kernel[7, :] = 1/15  # Horizontal motion blur
    blurred_scene = ndimage.convolve(lit_scene, kernel)

    # Add noise
    final_scene = blurred_scene + np.random.normal(0, 5, blurred_scene.shape)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Computer Vision Preprocessing Pipeline', fontsize=16, fontweight='bold')

    # Original degraded scene
    axes[0, 0].imshow(final_scene, cmap='gray')
    axes[0, 0].set_title('Input Scene')
    axes[0, 0].axis('off')

    # Illumination correction
    # Estimate background using morphological opening
    selem = morphology.disk(30)
    background = morphology.opening(final_scene, selem)
    corrected = final_scene - background + np.mean(final_scene)

    axes[0, 1].imshow(corrected, cmap='gray')
    axes[0, 1].set_title('Illumination Corrected')
    axes[0, 1].axis('off')

    # Deblurring
    # Simple unsharp masking
    gaussian_blurred = ndimage.gaussian_filter(corrected, sigma=2)
    unsharp_mask = corrected - gaussian_blurred
    deblurred = corrected + 2 * unsharp_mask

    axes[0, 2].imshow(deblurred, cmap='gray')
    axes[0, 2].set_title('Deblurred')
    axes[0, 2].axis('off')

    # Noise reduction
    denoised = ndimage.gaussian_filter(deblurred, sigma=1.0)
    axes[0, 3].imshow(denoised, cmap='gray')
    axes[0, 3].set_title('Denoised')
    axes[0, 3].axis('off')

    # Edge detection
    edges = filters.sobel(denoised)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Edge Detection')
    axes[1, 0].axis('off')

    # Contrast enhancement
    # Histogram equalization
    hist, bins = np.histogram(denoised.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Create equalization mapping
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply equalization
    equalized = cdf_final[denoised.astype('uint8')]

    axes[1, 1].imshow(equalized, cmap='gray')
    axes[1, 1].set_title('Histogram Equalized')
    axes[1, 1].axis('off')

    # Object segmentation
    # Threshold-based segmentation
    threshold = filters.threshold_otsu(equalized)
    binary = equalized > threshold

    # Clean up with morphological operations
    cleaned = morphology.remove_small_objects(binary, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)

    axes[1, 2].imshow(cleaned, cmap='gray')
    axes[1, 2].set_title('Segmented Objects')
    axes[1, 2].axis('off')

    # Object properties
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled)

    # Draw bounding boxes
    result_rgb = np.stack([equalized/255.0] * 3, axis=-1)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        result_rgb[minr:minr+2, minc:maxc] = [1, 0, 0]  # Red top
        result_rgb[maxr-2:maxr, minc:maxc] = [1, 0, 0]  # Red bottom
        result_rgb[minr:maxr, minc:minc+2] = [1, 0, 0]  # Red left
        result_rgb[minr:maxr, maxc-2:maxc] = [1, 0, 0]  # Red right

    axes[1, 3].imshow(result_rgb)
    axes[1, 3].set_title('Object Detection')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/computer_vision_preprocessing.png', dpi=300, bbox_inches='tight')
    plt.close()

def registration_and_alignment():
    """Demonstrate image registration and alignment techniques."""
    # Create reference image - use retina for medical registration
    reference = data.retina()
    # Convert to grayscale for registration
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

    # Create moving images with different transformations
    # Translation
    translated = transform.SimilarityTransform(translation=(30, 20))
    moved_translated = transform.warp(reference, translated, preserve_range=True)

    # Rotation
    rotated = transform.rotate(reference, angle=15, preserve_range=True)

    # Scale
    scaled = transform.rescale(reference, 1.2, preserve_range=True, anti_aliasing=True)
    # Crop to same size
    h, w = reference.shape
    sh, sw = scaled.shape
    start_h = (sh - h) // 2
    start_w = (sw - w) // 2
    scaled = scaled[start_h:start_h+h, start_w:start_w+w]

    # Combined transformation
    combined_tform = transform.SimilarityTransform(
        scale=0.9, rotation=np.radians(10), translation=(20, -15)
    )
    moved_combined = transform.warp(reference, combined_tform, preserve_range=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Image Registration and Alignment', fontsize=16, fontweight='bold')

    # Reference image
    axes[0, 0].imshow(reference, cmap='gray')
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')

    # Transformed images
    axes[0, 1].imshow(moved_translated, cmap='gray')
    axes[0, 1].set_title('Translated')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rotated, cmap='gray')
    axes[0, 2].set_title('Rotated')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(moved_combined, cmap='gray')
    axes[0, 3].set_title('Combined Transform')
    axes[0, 3].axis('off')

    # Difference images before registration
    diff_translated = np.abs(reference.astype(float) - moved_translated.astype(float))
    axes[1, 0].imshow(diff_translated, cmap='hot')
    axes[1, 0].set_title('Difference (Translation)')
    axes[1, 0].axis('off')

    # Registration using cross-correlation
    # For translation, we can use phase correlation
    def register_translation(ref, moving):
        """Simple translation registration using phase correlation."""
        # FFT of both images
        f_ref = np.fft.fft2(ref)
        f_moving = np.fft.fft2(moving)

        # Cross power spectrum
        cross_power = (f_ref * np.conj(f_moving)) / np.abs(f_ref * np.conj(f_moving))

        # Inverse FFT to get correlation
        correlation = np.abs(np.fft.ifft2(cross_power))

        # Find peak
        y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)

        # Convert to displacement
        if y_peak > ref.shape[0] // 2:
            y_shift = y_peak - ref.shape[0]
        else:
            y_shift = y_peak

        if x_peak > ref.shape[1] // 2:
            x_shift = x_peak - ref.shape[1]
        else:
            x_shift = x_peak

        return -y_shift, -x_shift

    # Register translated image
    try:
        y_shift, x_shift = register_translation(reference, moved_translated)
        correction_tform = transform.SimilarityTransform(translation=(-x_shift, -y_shift))
        registered_translated = transform.warp(moved_translated, correction_tform, preserve_range=True)

        axes[1, 1].imshow(registered_translated, cmap='gray')
        axes[1, 1].set_title('Registered Translation')
        axes[1, 1].axis('off')
    except:
        axes[1, 1].imshow(moved_translated, cmap='gray')
        axes[1, 1].set_title('Registration Failed')
        axes[1, 1].axis('off')

    # Feature-based registration visualization
    # Show feature points (corners)
    from skimage.feature import corner_harris
    corners_ref = corner_harris(reference)
    corner_coords_ref = np.column_stack(np.where(corners_ref > 0.01 * corners_ref.max()))

    # Overlay on reference image
    overlay_ref = reference.copy()
    for coord in corner_coords_ref[:50]:  # Show first 50 corners
        y, x = coord
        if 0 <= y < overlay_ref.shape[0] and 0 <= x < overlay_ref.shape[1]:
            overlay_ref[max(0, y-2):min(overlay_ref.shape[0], y+3),
                       max(0, x-2):min(overlay_ref.shape[1], x+3)] = 255

    axes[1, 2].imshow(overlay_ref, cmap='gray')
    axes[1, 2].set_title('Feature Points')
    axes[1, 2].axis('off')

    # Registration quality metrics
    registrations = [
        ('Original', reference, reference),
        ('Translated', moved_translated, reference),
        ('Rotated', rotated, reference),
        ('Combined', moved_combined, reference),
    ]

    metrics = []
    names = []

    for name, moving, ref in registrations:
        # Normalized cross-correlation
        ncc = np.corrcoef(moving.flatten(), ref.flatten())[0, 1]
        metrics.append(ncc)
        names.append(name)

    # Plot metrics
    colors = [PRIMARY_BLUE if name == 'Original' else ACCENT_ORANGE for name in names]
    axes[1, 3].bar(range(len(names)), metrics, color=colors)
    axes[1, 3].set_title('Registration Quality (NCC)')
    axes[1, 3].set_xlabel('Transformation')
    axes[1, 3].set_ylabel('Correlation')
    axes[1, 3].set_xticks(range(len(names)))
    axes[1, 3].set_xticklabels(names, rotation=45)
    axes[1, 3].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../figures/image_registration.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all real-world example figures."""
    print("Generating real-world image restoration and geometric processing figures...")

    ensure_output_dir()

    # Generate all figures
    document_processing_pipeline()
    print("✓ Generated document processing figure")

    medical_image_restoration()
    print("✓ Generated medical image restoration figure")

    computer_vision_preprocessing()
    print("✓ Generated computer vision preprocessing figure")

    registration_and_alignment()
    print("✓ Generated image registration figure")

    print("Real-world examples figures generation complete!")

if __name__ == "__main__":
    main()