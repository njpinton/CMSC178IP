"""
Real-World Feature Extraction Examples
Demonstrates practical applications of feature extraction in real scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature, filters, morphology, measure, segmentation
from skimage.data import camera, coins, coffee
from scipy import ndimage
import matplotlib.patches as patches

# Set matplotlib style for consistency
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_document_analysis():
    """Demonstrate feature extraction for document analysis."""
    # Create a synthetic document image
    document = np.ones((400, 600), dtype=np.uint8) * 255

    # Add text lines (simplified as rectangles)
    text_lines = [
        (50, 50, 500, 20),   # Title
        (50, 100, 400, 15),  # Line 1
        (50, 130, 450, 15),  # Line 2
        (50, 160, 380, 15),  # Line 3
        (50, 220, 420, 15),  # Line 4
        (50, 250, 390, 15),  # Line 5
        (50, 310, 200, 80),  # Image placeholder
        (280, 310, 270, 80), # Text column
    ]

    for x, y, w, h in text_lines:
        if w > 300:  # Title or long lines
            document[y:y+h, x:x+w] = 100
        elif h > 50:  # Image placeholder
            document[y:y+h, x:x+w] = 150
        else:  # Regular text
            document[y:y+h, x:x+w] = 80

    # Add some noise
    noise = np.random.normal(0, 10, document.shape)
    document = np.clip(document.astype(float) + noise, 0, 255).astype(np.uint8)

    # Apply feature extraction techniques
    edges = feature.canny(document, sigma=1, low_threshold=50, high_threshold=150)

    # Horizontal and vertical line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, vertical_kernel)

    # Text region detection using morphology
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    text_regions = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, text_kernel)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original document
    axes[0, 0].imshow(document, cmap='gray')
    axes[0, 0].set_title('Original Document')
    axes[0, 0].axis('off')

    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    axes[0, 1].axis('off')

    # Horizontal lines
    axes[0, 2].imshow(horizontal_lines, cmap='gray')
    axes[0, 2].set_title('Horizontal Line Detection')
    axes[0, 2].axis('off')

    # Vertical lines
    axes[1, 0].imshow(vertical_lines, cmap='gray')
    axes[1, 0].set_title('Vertical Line Detection')
    axes[1, 0].axis('off')

    # Text regions
    axes[1, 1].imshow(text_regions, cmap='gray')
    axes[1, 1].set_title('Text Region Detection')
    axes[1, 1].axis('off')

    # Layout analysis
    layout_analysis = cv2.cvtColor(document, cv2.COLOR_GRAY2RGB)

    # Find contours for layout analysis
    contours, _ = cv2.findContours(text_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 30:  # Text line
                cv2.rectangle(layout_analysis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:  # Image or paragraph
                cv2.rectangle(layout_analysis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    axes[1, 2].imshow(layout_analysis)
    axes[1, 2].set_title('Layout Analysis')
    axes[1, 2].axis('off')

    plt.suptitle('Document Analysis using Feature Extraction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/document_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_medical_imaging():
    """Demonstrate feature extraction for medical imaging applications."""
    # Create synthetic medical images (X-ray like)
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Create a chest X-ray like image
    chest = np.ones((size, size))

    # Ribcage pattern
    for i in range(5):
        rib_y = -0.6 + i * 0.3
        rib_mask = (np.abs(Y - rib_y) < 0.05) & (np.abs(X) < 0.7)
        chest[rib_mask] = 0.3

    # Lung regions
    left_lung = ((X + 0.3)**2 + Y**2 < 0.25) & (X < 0)
    right_lung = ((X - 0.3)**2 + Y**2 < 0.25) & (X > 0)
    chest[left_lung | right_lung] = 0.7

    # Add some abnormalities (spots)
    spot1 = ((X + 0.2)**2 + (Y - 0.1)**2 < 0.02)
    spot2 = ((X - 0.15)**2 + (Y + 0.2)**2 < 0.015)
    chest[spot1 | spot2] = 0.2

    # Add noise
    noise = np.random.normal(0, 0.05, chest.shape)
    chest = np.clip(chest + noise, 0, 1)

    # Apply feature extraction
    edges = feature.canny(chest, sigma=1.5)
    blobs_log = feature.blob_log(chest, min_sigma=5, max_sigma=20, num_sigma=10, threshold=0.1)

    # Texture analysis using LBP
    chest_uint8 = (chest * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(chest_uint8, P=8, R=2, method='uniform')

    # Region growing for lung segmentation
    # Simple threshold-based segmentation
    lung_mask = (chest > 0.5) & (chest < 0.8)
    lung_mask = morphology.binary_opening(lung_mask, morphology.disk(5))
    lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(chest, cmap='gray')
    axes[0, 0].set_title('Synthetic Chest X-ray')
    axes[0, 0].axis('off')

    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection (Canny)')
    axes[0, 1].axis('off')

    # Blob detection
    axes[0, 2].imshow(chest, cmap='gray')
    for blob in blobs_log:
        y, x, r = blob
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        axes[0, 2].add_patch(circle)
    axes[0, 2].set_title(f'Blob Detection ({len(blobs_log)} blobs)')
    axes[0, 2].axis('off')

    # Texture analysis
    axes[1, 0].imshow(lbp, cmap='gray')
    axes[1, 0].set_title('Local Binary Pattern')
    axes[1, 0].axis('off')

    # Lung segmentation
    axes[1, 1].imshow(lung_mask, cmap='gray')
    axes[1, 1].set_title('Lung Region Segmentation')
    axes[1, 1].axis('off')

    # Combined analysis
    combined = cv2.cvtColor((chest * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    combined[lung_mask] = [0, 255, 0]  # Green for lungs
    for blob in blobs_log:
        y, x, r = blob
        cv2.circle(combined, (int(x), int(y)), int(r), (255, 0, 0), 2)

    axes[1, 2].imshow(combined)
    axes[1, 2].set_title('Combined Analysis')
    axes[1, 2].axis('off')

    plt.suptitle('Medical Imaging Feature Extraction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/medical_imaging.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_industrial_inspection():
    """Demonstrate feature extraction for industrial quality inspection."""
    # Create synthetic PCB (Printed Circuit Board) image
    pcb = np.ones((300, 400), dtype=np.float32) * 0.2  # Dark background

    # Add circuit traces
    traces = [
        (50, 50, 300, 52),   # Horizontal trace
        (50, 100, 300, 102), # Horizontal trace
        (100, 50, 102, 200), # Vertical trace
        (200, 50, 202, 200), # Vertical trace
    ]

    for x1, y1, x2, y2 in traces:
        cv2.rectangle(pcb, (x1, y1), (x2, y2), 0.8, -1)

    # Add components (resistors, capacitors)
    components = [
        (120, 80, 160, 95),   # Resistor 1
        (220, 80, 260, 95),   # Resistor 2
        (120, 120, 140, 140), # Capacitor 1
        (180, 120, 200, 140), # Capacitor 2
    ]

    for x1, y1, x2, y2 in components:
        cv2.rectangle(pcb, (x1, y1), (x2, y2), 1.0, -1)

    # Add defects
    # Missing component
    missing_component = (260, 120, 280, 140)
    # Broken trace
    cv2.rectangle(pcb, (150, 100), (170, 102), 0.2, -1)

    # Add solder joints
    solder_joints = [
        (120, 85), (160, 85), (220, 85), (260, 85),  # Component ends
        (125, 125), (135, 125), (185, 125), (195, 125)
    ]

    for x, y in solder_joints:
        cv2.circle(pcb, (x, y), 3, 0.9, -1)

    # Add noise
    noise = np.random.normal(0, 0.02, pcb.shape)
    pcb = np.clip(pcb + noise, 0, 1)

    # Feature extraction for inspection
    edges = feature.canny(pcb, sigma=1.0)

    # Template matching for components
    component_template = np.ones((15, 40), dtype=np.uint8) * 255
    pcb_uint8_for_template = (pcb * 255).astype(np.uint8)
    template_match = cv2.matchTemplate(pcb_uint8_for_template, component_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(template_match >= 0.3)

    # Contour detection for shape analysis
    pcb_uint8 = (pcb * 255).astype(np.uint8)
    _, binary = cv2.threshold(pcb_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Defect detection using morphological operations
    # Detect breaks in traces
    trace_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    trace_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, trace_kernel)
    trace_defects = cv2.subtract(trace_close, binary)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original PCB
    axes[0, 0].imshow(pcb, cmap='gray')
    axes[0, 0].set_title('PCB Image')
    axes[0, 0].axis('off')

    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    axes[0, 1].axis('off')

    # Template matching
    axes[0, 2].imshow(template_match, cmap='hot')
    axes[0, 2].set_title('Component Template Matching')
    axes[0, 2].axis('off')

    # Binary segmentation
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Segmentation')
    axes[1, 0].axis('off')

    # Defect detection
    axes[1, 1].imshow(trace_defects, cmap='gray')
    axes[1, 1].set_title('Trace Defect Detection')
    axes[1, 1].axis('off')

    # Quality inspection result
    inspection_result = cv2.cvtColor(pcb_uint8, cv2.COLOR_GRAY2RGB)

    # Mark detected components
    for pt in zip(*locations[::-1]):
        cv2.rectangle(inspection_result, pt, (pt[0] + 40, pt[1] + 15), (0, 255, 0), 2)

    # Mark defects
    defect_contours, _ = cv2.findContours(trace_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in defect_contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(inspection_result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mark missing component area
    cv2.rectangle(inspection_result, missing_component[:2], missing_component[2:], (255, 0, 0), 2)
    cv2.putText(inspection_result, 'MISSING', (missing_component[0], missing_component[1]-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    axes[1, 2].imshow(inspection_result)
    axes[1, 2].set_title('Quality Inspection Result')
    axes[1, 2].axis('off')

    plt.suptitle('Industrial PCB Quality Inspection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/industrial_inspection.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_biometric_analysis():
    """Demonstrate feature extraction for biometric applications."""
    # Create synthetic fingerprint pattern
    size = 200
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Create fingerprint-like ridges
    fingerprint = np.zeros((size, size))

    # Concentric elliptical patterns
    for i in range(10):
        a = 0.1 + i * 0.08  # Semi-major axis
        b = 0.08 + i * 0.06  # Semi-minor axis
        ellipse = (X**2 / a**2 + Y**2 / b**2)
        ridge_mask = (ellipse > 0.8) & (ellipse < 1.0)
        fingerprint[ridge_mask] = 1

    # Add some noise and blur
    noise = np.random.normal(0, 0.1, fingerprint.shape)
    fingerprint = ndimage.gaussian_filter(fingerprint + noise, sigma=0.5)
    fingerprint = np.clip(fingerprint, 0, 1)

    # Feature extraction for fingerprint
    # Ridge detection using oriented filters
    angles = [0, 30, 60, 90, 120, 150]
    gabor_responses = []

    for angle in angles:
        theta = np.radians(angle)
        gabor_real, _ = filters.gabor(fingerprint, frequency=0.3, theta=theta)
        gabor_responses.append(gabor_real)

    # Minutiae detection (simplified)
    edges = feature.canny(fingerprint, sigma=1.0)

    # Ridge ending and bifurcation detection using morphological operations
    skeleton = morphology.skeletonize(fingerprint > 0.5)

    # Find minutiae points (simplified approach)
    minutiae_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    minutiae_response = cv2.filter2D(skeleton.astype(np.uint8), -1, minutiae_kernel)

    # Ridge endings (1 neighbor) and bifurcations (3+ neighbors)
    ridge_endings = (minutiae_response == 1) & skeleton
    bifurcations = (minutiae_response >= 3) & skeleton

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original fingerprint
    axes[0, 0].imshow(fingerprint, cmap='gray')
    axes[0, 0].set_title('Synthetic Fingerprint')
    axes[0, 0].axis('off')

    # Gabor filter responses
    gabor_combined = np.maximum.reduce(gabor_responses)
    axes[0, 1].imshow(gabor_combined, cmap='gray')
    axes[0, 1].set_title('Gabor Filter Response')
    axes[0, 1].axis('off')

    # Ridge skeleton
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Ridge Skeleton')
    axes[0, 2].axis('off')

    # Minutiae detection
    minutiae_image = cv2.cvtColor((fingerprint * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Mark ridge endings in red
    ending_points = np.where(ridge_endings)
    for y, x in zip(ending_points[0], ending_points[1]):
        cv2.circle(minutiae_image, (x, y), 3, (255, 0, 0), -1)

    # Mark bifurcations in blue
    bifurcation_points = np.where(bifurcations)
    for y, x in zip(bifurcation_points[0], bifurcation_points[1]):
        cv2.circle(minutiae_image, (x, y), 3, (0, 0, 255), -1)

    axes[1, 0].imshow(minutiae_image)
    axes[1, 0].set_title(f'Minutiae Detection\nEndings: {len(ending_points[0])}, Bifurcations: {len(bifurcation_points[0])}')
    axes[1, 0].axis('off')

    # Orientation field
    block_size = 16
    orientation_field = np.zeros((size // block_size, size // block_size))

    for i in range(0, size - block_size, block_size):
        for j in range(0, size - block_size, block_size):
            block = fingerprint[i:i+block_size, j:j+block_size]

            # Compute gradients
            gx = ndimage.sobel(block, axis=1)
            gy = ndimage.sobel(block, axis=0)

            # Compute orientation
            orientation = np.arctan2(np.sum(gy), np.sum(gx))
            orientation_field[i//block_size, j//block_size] = orientation

    im = axes[1, 1].imshow(orientation_field, cmap='hsv')
    axes[1, 1].set_title('Ridge Orientation Field')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Quality assessment
    quality_map = ndimage.gaussian_filter(gabor_combined**2, sigma=2)
    axes[1, 2].imshow(quality_map, cmap='viridis')
    axes[1, 2].set_title('Fingerprint Quality Map')
    axes[1, 2].axis('off')

    plt.suptitle('Biometric Fingerprint Feature Extraction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/biometric_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_autonomous_vehicle():
    """Demonstrate feature extraction for autonomous vehicle applications."""
    # Create synthetic road scene
    road_scene = np.ones((300, 400, 3), dtype=np.uint8) * 135  # Sky color

    # Road
    road_points = np.array([[0, 200], [400, 200], [350, 300], [50, 300]], np.int32)
    cv2.fillPoly(road_scene, [road_points], (80, 80, 80))

    # Lane markings
    for i in range(0, 400, 40):
        cv2.rectangle(road_scene, (i, 240), (i+20, 245), (255, 255, 255), -1)

    # Vehicles
    cv2.rectangle(road_scene, (150, 180), (200, 210), (255, 0, 0), -1)  # Red car
    cv2.rectangle(road_scene, (280, 190), (320, 215), (0, 255, 0), -1)  # Green car

    # Traffic signs
    cv2.circle(road_scene, (50, 150), 15, (255, 255, 0), -1)  # Yellow sign
    cv2.rectangle(road_scene, (350, 140), (380, 170), (255, 0, 255), -1)  # Magenta sign

    # Trees/obstacles
    cv2.circle(road_scene, (30, 120), 20, (0, 128, 0), -1)
    cv2.circle(road_scene, (370, 130), 18, (0, 128, 0), -1)

    # Convert to grayscale for processing
    gray_scene = cv2.cvtColor(road_scene, cv2.COLOR_RGB2GRAY)

    # Lane detection using Hough transform
    edges = feature.canny(gray_scene, sigma=1, low_threshold=50, high_threshold=150)
    edges_uint8 = (edges * 255).astype(np.uint8)
    lines = cv2.HoughLinesP(edges_uint8, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

    # Vehicle detection using template matching
    vehicle_template = np.ones((30, 50), dtype=np.uint8) * 128
    match_result = cv2.matchTemplate(gray_scene, vehicle_template, cv2.TM_CCOEFF_NORMED)
    vehicle_locations = np.where(match_result >= 0.3)

    # Object detection using contours
    _, binary = cv2.threshold(gray_scene, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original scene
    axes[0, 0].imshow(road_scene)
    axes[0, 0].set_title('Road Scene')
    axes[0, 0].axis('off')

    # Edge detection
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection')
    axes[0, 1].axis('off')

    # Lane detection
    lane_image = road_scene.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter for lane-like lines (roughly horizontal in perspective)
            if abs(y2 - y1) < 20:  # Horizontal-ish lines
                cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 255), 3)

    axes[0, 2].imshow(lane_image)
    axes[0, 2].set_title('Lane Detection')
    axes[0, 2].axis('off')

    # Vehicle detection
    vehicle_image = road_scene.copy()
    for pt in zip(*vehicle_locations[::-1]):
        cv2.rectangle(vehicle_image, pt, (pt[0] + 50, pt[1] + 30), (255, 255, 0), 2)

    axes[1, 0].imshow(vehicle_image)
    axes[1, 0].set_title('Vehicle Detection')
    axes[1, 0].axis('off')

    # Object segmentation
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title('Object Segmentation')
    axes[1, 1].axis('off')

    # Complete analysis
    analysis_result = road_scene.copy()

    # Draw detected objects with classification
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if aspect_ratio > 1.5:  # Wide objects (vehicles)
                cv2.rectangle(analysis_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(analysis_result, 'Vehicle', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif area < 1000:  # Small objects (signs)
                cv2.rectangle(analysis_result, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(analysis_result, 'Sign', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:  # Other objects
                cv2.rectangle(analysis_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(analysis_result, 'Object', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw lane lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20:
                cv2.line(analysis_result, (x1, y1), (x2, y2), (255, 255, 255), 2)

    axes[1, 2].imshow(analysis_result)
    axes[1, 2].set_title('Complete Scene Analysis')
    axes[1, 2].axis('off')

    plt.suptitle('Autonomous Vehicle Vision System', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/autonomous_vehicle.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating real-world feature extraction examples...")

    print("Creating document analysis example...")
    create_document_analysis()

    print("Creating medical imaging example...")
    create_medical_imaging()

    print("Creating industrial inspection example...")
    create_industrial_inspection()

    print("Creating biometric analysis example...")
    create_biometric_analysis()

    print("Creating autonomous vehicle example...")
    create_autonomous_vehicle()

    print("Real-world examples figures generated successfully!")