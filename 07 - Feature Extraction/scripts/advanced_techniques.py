"""
Advanced Feature Extraction Techniques
Generates figures demonstrating SIFT, ORB, HOG, and other advanced feature descriptors.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature, transform, exposure
from skimage.data import camera, coins
# from sklearn.cluster import KMeans  # Not available, will use simple clustering
import matplotlib.patches as patches

# Set matplotlib style for consistency
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_sift_features():
    """Demonstrate SIFT feature detection and description."""
    # Load test image
    image = camera()

    # Convert to uint8 for OpenCV
    image_cv = (image).astype(np.uint8)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image_cv, keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create scale-space visualization
    scales = [1.0, 1.6, 2.56, 4.1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image with keypoints
    axes[0, 0].imshow(image_with_keypoints)
    axes[0, 0].set_title(f'SIFT Keypoints ({len(keypoints)} detected)')
    axes[0, 0].axis('off')

    # Show scale space
    for i, scale in enumerate(scales[:3]):
        sigma = scale
        blurred = cv2.GaussianBlur(image_cv, (0, 0), sigma)
        axes[0, i].imshow(blurred, cmap='gray')
        axes[0, i].set_title(f'Scale σ = {sigma:.1f}')
        axes[0, i].axis('off')

    # Keypoint properties analysis
    if len(keypoints) > 0:
        scales_detected = [kp.size for kp in keypoints]
        orientations = [kp.angle for kp in keypoints]
        responses = [kp.response for kp in keypoints]

        # Scale distribution
        axes[1, 0].hist(scales_detected, bins=20, alpha=0.7, color='blue')
        axes[1, 0].set_title('Keypoint Scale Distribution')
        axes[1, 0].set_xlabel('Scale')
        axes[1, 0].set_ylabel('Count')

        # Orientation distribution
        axes[1, 1].hist(orientations, bins=36, alpha=0.7, color='green')
        axes[1, 1].set_title('Keypoint Orientation Distribution')
        axes[1, 1].set_xlabel('Orientation (degrees)')
        axes[1, 1].set_ylabel('Count')

        # Response strength
        axes[1, 2].hist(responses, bins=20, alpha=0.7, color='red')
        axes[1, 2].set_title('Keypoint Response Strength')
        axes[1, 2].set_xlabel('Response')
        axes[1, 2].set_ylabel('Count')

    plt.suptitle('SIFT (Scale-Invariant Feature Transform)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/sift_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_orb_features():
    """Demonstrate ORB feature detection and description."""
    image = camera()
    image_cv = (image).astype(np.uint8)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image_cv, None)

    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image_cv, keypoints, None, color=(0,255,0))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # ORB keypoints
    axes[0, 1].imshow(image_with_keypoints)
    axes[0, 1].set_title(f'ORB Keypoints ({len(keypoints)} detected)')
    axes[0, 1].axis('off')

    if len(keypoints) > 0 and descriptors is not None:
        # Descriptor visualization
        # Take first 50 descriptors for visualization
        desc_subset = descriptors[:min(50, len(descriptors))]
        axes[1, 0].imshow(desc_subset, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('ORB Descriptors (Binary)')
        axes[1, 0].set_xlabel('Descriptor Dimension')
        axes[1, 0].set_ylabel('Keypoint Index')

        # Hamming distance matrix between descriptors
        if len(desc_subset) > 1:
            n_desc = min(20, len(desc_subset))
            hamming_matrix = np.zeros((n_desc, n_desc))
            for i in range(n_desc):
                for j in range(n_desc):
                    hamming_matrix[i, j] = cv2.norm(desc_subset[i], desc_subset[j], cv2.NORM_HAMMING)

            im = axes[1, 1].imshow(hamming_matrix, cmap='plasma')
            axes[1, 1].set_title('Hamming Distance Matrix')
            axes[1, 1].set_xlabel('Descriptor Index')
            axes[1, 1].set_ylabel('Descriptor Index')
            plt.colorbar(im, ax=axes[1, 1])

    plt.suptitle('ORB (Oriented FAST and Rotated BRIEF)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/orb_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hog_features():
    """Demonstrate HOG (Histogram of Oriented Gradients) features."""
    image = camera()

    # Compute HOG features
    hog_features, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                         cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

    # Create a simple object for demonstration (pedestrian-like shape)
    pedestrian = np.zeros((128, 64), dtype=np.uint8)
    # Head
    cv2.circle(pedestrian, (32, 20), 8, 255, -1)
    # Body
    cv2.rectangle(pedestrian, (28, 28), (36, 80), 255, -1)
    # Arms
    cv2.rectangle(pedestrian, (20, 35), (28, 55), 255, -1)
    cv2.rectangle(pedestrian, (36, 35), (44, 55), 255, -1)
    # Legs
    cv2.rectangle(pedestrian, (28, 80), (32, 120), 255, -1)
    cv2.rectangle(pedestrian, (32, 80), (36, 120), 255, -1)

    # Compute HOG for pedestrian
    ped_hog_features, ped_hog_image = feature.hog(pedestrian, orientations=9, pixels_per_cell=(8, 8),
                                                 cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original camera image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Camera Image')
    axes[0, 0].axis('off')

    # HOG visualization for camera
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    axes[0, 1].imshow(hog_image_rescaled, cmap='gray')
    axes[0, 1].set_title('HOG Features')
    axes[0, 1].axis('off')

    # HOG feature histogram
    axes[0, 2].hist(hog_features, bins=50, alpha=0.7, color='blue')
    axes[0, 2].set_title('HOG Feature Distribution')
    axes[0, 2].set_xlabel('Feature Value')
    axes[0, 2].set_ylabel('Count')

    # Pedestrian template
    axes[1, 0].imshow(pedestrian, cmap='gray')
    axes[1, 0].set_title('Pedestrian Template')
    axes[1, 0].axis('off')

    # Pedestrian HOG
    ped_hog_image_rescaled = exposure.rescale_intensity(ped_hog_image, in_range=(0, 10))
    axes[1, 1].imshow(ped_hog_image_rescaled, cmap='gray')
    axes[1, 1].set_title('Pedestrian HOG Features')
    axes[1, 1].axis('off')

    # Feature comparison
    axes[1, 2].plot(hog_features[:100], 'b-', alpha=0.7, label='Camera Image')
    axes[1, 2].plot(ped_hog_features[:100], 'r-', alpha=0.7, label='Pedestrian')
    axes[1, 2].set_title('Feature Comparison (First 100)')
    axes[1, 2].set_xlabel('Feature Index')
    axes[1, 2].set_ylabel('Feature Value')
    axes[1, 2].legend()

    plt.suptitle('HOG (Histogram of Oriented Gradients) Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/hog_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lbp_analysis():
    """Advanced Local Binary Pattern analysis."""
    image = camera()

    # Different LBP parameters
    methods = ['default', 'uniform', 'nri_uniform', 'var']
    radii = [1, 2, 3]
    n_points = [8, 16, 24]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Different methods with R=1, P=8
    for i, method in enumerate(methods[:3]):
        lbp = feature.local_binary_pattern(image, P=8, R=1, method=method)
        axes[0, i+1].imshow(lbp, cmap='gray')
        axes[0, i+1].set_title(f'LBP {method}')
        axes[0, i+1].axis('off')

    # Different radii with uniform method
    for i, radius in enumerate(radii):
        lbp = feature.local_binary_pattern(image, P=8, R=radius, method='uniform')
        axes[1, i].imshow(lbp, cmap='gray')
        axes[1, i].set_title(f'LBP R={radius}')
        axes[1, i].axis('off')

    # Histogram comparison
    lbp_hist = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp_hist, bins=10)
    axes[1, 3].bar(range(len(hist)), hist, color='skyblue')
    axes[1, 3].set_title('LBP Histogram')
    axes[1, 3].set_xlabel('LBP Pattern')
    axes[1, 3].set_ylabel('Frequency')

    # Different number of points
    for i, points in enumerate(n_points):
        lbp = feature.local_binary_pattern(image, P=points, R=2, method='uniform')
        axes[2, i].imshow(lbp, cmap='gray')
        axes[2, i].set_title(f'LBP P={points}')
        axes[2, i].axis('off')

    # Texture discrimination example
    # Create two different texture regions
    texture1 = image[50:150, 50:150]  # Top-left region
    texture2 = image[200:300, 200:300]  # Bottom-right region

    lbp1 = feature.local_binary_pattern(texture1, P=8, R=1, method='uniform')
    lbp2 = feature.local_binary_pattern(texture2, P=8, R=1, method='uniform')

    hist1, _ = np.histogram(lbp1, bins=10, range=(0, 9))
    hist2, _ = np.histogram(lbp2, bins=10, range=(0, 9))

    x = np.arange(len(hist1))
    width = 0.35
    axes[2, 3].bar(x - width/2, hist1, width, label='Region 1', alpha=0.7)
    axes[2, 3].bar(x + width/2, hist2, width, label='Region 2', alpha=0.7)
    axes[2, 3].set_title('Texture Discrimination')
    axes[2, 3].set_xlabel('LBP Pattern')
    axes[2, 3].set_ylabel('Frequency')
    axes[2, 3].legend()

    plt.suptitle('Advanced Local Binary Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/advanced_lbp.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_matching():
    """Demonstrate feature matching between images."""
    # Load and prepare two similar images
    image1 = camera()

    # Create a slightly rotated and scaled version
    tform = transform.AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(10, 20))
    image2 = transform.warp(image1, tform.inverse, output_shape=image1.shape)
    image2 = (image2 * 255).astype(np.uint8)
    image1_cv = image1.astype(np.uint8)
    image2_cv = (image2 * 255).astype(np.uint8)

    # SIFT feature matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_cv, None)
    kp2, des2 = sift.detectAndCompute(image2_cv, None)

    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if des1 is not None and des2 is not None:
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original images
    axes[0, 0].imshow(image1, cmap='gray')
    axes[0, 0].set_title('Image 1 (Original)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2, cmap='gray')
    axes[0, 1].set_title('Image 2 (Transformed)')
    axes[0, 1].axis('off')

    # Keypoints
    img1_kp = cv2.drawKeypoints(image1_cv, kp1, None, color=(0,255,0))
    img2_kp = cv2.drawKeypoints(image2_cv, kp2, None, color=(0,255,0))

    axes[1, 0].imshow(img1_kp)
    axes[1, 0].set_title(f'Keypoints 1 ({len(kp1)})')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img2_kp)
    axes[1, 1].set_title(f'Keypoints 2 ({len(kp2)})')
    axes[1, 1].axis('off')

    plt.suptitle(f'Feature Matching - {len(good_matches)} good matches found', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/feature_matching.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate matching visualization
    if len(good_matches) > 0:
        # Draw matches
        img_matches = cv2.drawMatches(image1_cv, kp1, image2_cv, kp2, good_matches[:20], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 8))
        plt.imshow(img_matches)
        plt.title(f'Top 20 Feature Matches (Total: {len(good_matches)})')
        plt.axis('off')
        plt.savefig('../figures/feature_matches_lines.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_bag_of_features():
    """Demonstrate Bag of Visual Words concept."""
    # Create synthetic image patches with different patterns
    patch_size = 32
    n_patches = 100

    # Generate different types of patches
    patches = []
    labels = []

    # Edge patches
    for _ in range(25):
        patch = np.zeros((patch_size, patch_size))
        start_row = np.random.randint(0, patch_size//2)
        patch[start_row:start_row+2, :] = 1
        patches.append(patch)
        labels.append(0)

    # Corner patches
    for _ in range(25):
        patch = np.zeros((patch_size, patch_size))
        split = patch_size // 2
        patch[:split, :split] = 1
        patches.append(patch)
        labels.append(1)

    # Texture patches
    for _ in range(25):
        patch = np.random.random((patch_size, patch_size)) > 0.5
        patches.append(patch.astype(float))
        labels.append(2)

    # Blob patches
    for _ in range(25):
        patch = np.zeros((patch_size, patch_size))
        center = (patch_size//2, patch_size//2)
        y, x = np.ogrid[:patch_size, :patch_size]
        mask = (x - center[0])**2 + (y - center[1])**2 <= (patch_size//4)**2
        patch[mask] = 1
        patches.append(patch)
        labels.append(3)

    # Compute HOG features for each patch
    features = []
    for patch in patches:
        hog_feat = feature.hog(patch, orientations=8, pixels_per_cell=(8, 8),
                              cells_per_block=(1, 1), visualize=False)
        features.append(hog_feat)

    features = np.array(features)

    # Simple clustering by grouping similar features
    n_clusters = 8
    # Simple clustering: group features by their mean values
    feature_means = np.mean(features, axis=1)
    cluster_labels = np.digitize(feature_means, np.linspace(feature_means.min(), feature_means.max(), n_clusters)) - 1
    cluster_labels = np.clip(cluster_labels, 0, n_clusters-1)

    # Create pseudo cluster centers for visualization
    cluster_centers = np.zeros((n_clusters, features.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(features[mask], axis=0)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Show example patches from each category
    categories = ['Edges', 'Corners', 'Texture', 'Blobs']
    colors = ['red', 'green', 'blue', 'orange']

    for i, (category, color) in enumerate(zip(categories, colors)):
        # Find patches of this category
        category_indices = [j for j, label in enumerate(labels) if label == i]
        example_patch = patches[category_indices[0]]

        axes[0, i].imshow(example_patch, cmap='gray')
        axes[0, i].set_title(f'{category} Example')
        axes[0, i].axis('off')

    # Show cluster centers as visual words
    for i in range(min(8, n_clusters)):
        cluster_center = cluster_centers[i]
        # Reshape back to patch-like visualization (approximate)
        center_vis = cluster_center.reshape(-1, 1)

        axes[1, i % 4].imshow(center_vis, cmap='viridis', aspect='auto')
        axes[1, i % 4].set_title(f'Visual Word {i+1}')
        axes[1, i % 4].axis('off')

    # Show cluster assignments
    for i, color in enumerate(colors):
        category_clusters = [cluster_labels[j] for j, label in enumerate(labels) if label == i]
        hist, bins = np.histogram(category_clusters, bins=n_clusters, range=(0, n_clusters))

        axes[2, i].bar(range(n_clusters), hist, color=color, alpha=0.7)
        axes[2, i].set_title(f'{categories[i]} Word Distribution')
        axes[2, i].set_xlabel('Visual Word')
        axes[2, i].set_ylabel('Count')

    plt.suptitle('Bag of Visual Words Concept', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bag_of_features.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating advanced feature extraction visualizations...")

    print("Creating SIFT features demonstration...")
    create_sift_features()

    print("Creating ORB features demonstration...")
    create_orb_features()

    print("Creating HOG features demonstration...")
    create_hog_features()

    print("Creating advanced LBP analysis...")
    create_lbp_analysis()

    print("Creating feature matching demonstration...")
    create_feature_matching()

    print("Creating bag of features demonstration...")
    create_bag_of_features()

    print("Advanced techniques figures generated successfully!")