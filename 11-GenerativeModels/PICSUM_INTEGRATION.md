# Picsum Integration for Generative Models

## Summary

The Generative Models module has been enhanced to use **Picsum.photos** (Lorem Picsum) for fetching realistic placeholder images. This provides more diverse and professional-looking examples for demonstrating generative model applications, particularly for style transfer and image-to-image translation.

## What is Picsum?

[Picsum.photos](https://picsum.photos) is a free service that provides placeholder images for development and education purposes. It offers:
- High-quality, royalty-free images
- Specific image IDs for reproducibility
- On-the-fly image resizing
- Grayscale conversion option
- Simple API with no authentication required

## Changes Made

### 1. Updated `scripts/generative_applications.py`

#### New Function: `fetch_picsum_image()`
```python
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
```

**Features:**
- Fetches images from Picsum API
- Supports specific image IDs for reproducibility
- Automatic fallback to scikit-image datasets if Picsum is unavailable
- Handles both color and grayscale images
- Returns normalized numpy arrays ready for processing

#### Updated Functions:

1. **`create_style_transfer_concept()`**
   - Now uses Picsum images instead of just `skimage.data.camera()`
   - Uses specific image IDs for reproducible results
   - Demonstrates style transfer with diverse, realistic images

2. **`create_image_to_image_translation()`** (NEW)
   - Demonstrates image-to-image translation (e.g., pix2pix, CycleGAN)
   - Shows color-to-grayscale translation as an example
   - Uses multiple diverse Picsum images
   - Creates professional visualization with domain labels

### 2. Updated `requirements.txt`

Added:
```
# For fetching images from Picsum
requests>=2.25.0
```

### 3. Updated README.md

Added documentation:
- Noted that some figures use Picsum.photos
- Explained internet connection requirement
- Documented fallback behavior when Picsum is unavailable

## Usage

The script will automatically fetch images from Picsum when generating figures:

```bash
cd 11-GenerativeModels/scripts
python generative_applications.py
```

Or with the virtual environment:
```bash
cd 11-GenerativeModels
source venv/bin/activate
python scripts/generative_applications.py
```

## Image IDs Used

For reproducibility, the following specific Picsum image IDs are used:

- **Style Transfer**:
  - Content: ID 1015 (landscape/architecture)
  - Style: ID 1018 (textured image)

- **Image-to-Image Translation**:
  - IDs: 1025, 1043, 1047, 1050 (diverse scene types)

These IDs can be viewed at: `https://picsum.photos/id/{ID}/info`

## Fallback Mechanism

If Picsum is unavailable (no internet, API down, timeout), the script automatically falls back to:
- `skimage.data.camera()` for single images
- Maintains full functionality without internet

Example fallback message:
```
Warning: Could not fetch from Picsum (timeout), using fallback image
```

## Benefits

1. **Realistic Examples**: Real photographs instead of synthetic data
2. **Reproducibility**: Specific image IDs ensure consistent results
3. **Diversity**: Access to thousands of different images
4. **Robustness**: Fallback ensures script always works
5. **Educational**: Better demonstrates real-world applications

## Generated Figures

The following figures now use or benefit from Picsum:

- `style_transfer_concept.png` - Uses Picsum images for content and style
- `image_to_image_translation.png` - NEW figure using multiple Picsum images

## Future Enhancements

Potential improvements:
1. Add color-based style transfer examples
2. Use Picsum for super-resolution demonstrations
3. Create image inpainting examples
4. Add more diverse image-to-image translation pairs

## Testing

The script was successfully tested and generated all figures, including:
- Generation examples
- Latent space interpolation
- Conditional generation
- Style transfer (with Picsum)
- Image-to-image translation (with Picsum)
- Applications overview
- Training tips

## Notes

- The script uses a 10-second timeout for Picsum requests
- All images are converted to float [0,1] range for consistency
- Grayscale conversion is handled automatically when requested
- The fallback mechanism is transparent to end users

---

**Date**: October 2025
**Status**: ✓ Implemented and Tested
