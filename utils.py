import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from PIL import Image

def save_detection_results(results, base_output_dir="results"):
    """
    Save detection results to a structured folder system
    
    Args:
        results: List of dictionaries containing detection results for each image
        base_output_dir: Base directory where to save all results
    
    Each image gets its own folder with:
    - anomaly patches saved as PNG
    - a text file with detection metadata
    - marked original image
    - segmentation mask image
    """
    # Create base output directory if it doesn't exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Process each image's results
    for result in results:
        # Get filename without extension to use as folder name
        image_name = os.path.splitext(result['filename'])[0]
        image_dir = os.path.join(base_output_dir, image_name)
        
        # Create image-specific directory
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # 1. Save metadata to text file
        metadata_path = os.path.join(image_dir, f"{image_name}_detections.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Anomaly detection results for {result['filename']}\n")
            f.write(f"Total anomalies detected: {len(result['anomalies'])}\n\n")
            
            # Write metadata for each anomaly
            for i, anomaly in enumerate(result['anomalies']):
                x, y = anomaly['coords']
                
                # Calculate bounding box (approximate from contour if available)
                if 'contour' in anomaly:
                    x, y, w, h = cv2.boundingRect(anomaly['contour'])
                    confidence = 1.0  # Default confidence when using contour-based detection
                else:
                    # Estimate from patch size
                    patch = anomaly['patch']
                    h, w = patch.shape[:2]
                    confidence = 0.9  # Default confidence for patch-based detection
                
                # Write metadata
                f.write(f"Anomaly {i+1}:\n")
                f.write(f"  Position: ({x}, {y})\n")
                f.write(f"  Bounding box: x={x}, y={y}, width={w}, height={h}\n")
                f.write(f"  Confidence: {confidence:.2f}\n\n")
        
        # 2. Save anomaly patches
        for i, anomaly in enumerate(result['anomalies']):
            patch = anomaly['patch']
            patch_path = os.path.join(image_dir, f"{image_name}_anomaly_{i+1}.png")
            
            # Convert to PIL Image and save
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if len(patch.shape) == 3 else patch
            Image.fromarray(patch_rgb).save(patch_path)
        
        # 3. Save original image with anomalies marked
        plt.figure(figsize=(10, 8))
        plt.imshow(result['image'])
        plt.axis('off')
        
        # Mark anomalies with red X
        for anomaly in result['anomalies']:
            plt.plot(anomaly['coords'][0], anomaly['coords'][1], 'rx', markersize=10, linewidth=3)
        
        # Save marked image
        marked_image_path = os.path.join(image_dir, f"{image_name}_marked.png")
        plt.tight_layout(pad=0)
        plt.savefig(marked_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 4. Save segmentation mask visualization
        if 'vis_mask' in result:
            # If we have a visualization mask (colored version)
            vis_mask_path = os.path.join(image_dir, f"{image_name}_segmentation_vis.png")
            Image.fromarray(result['vis_mask']).save(vis_mask_path)
        
        # Save binary segmentation mask
        mask_path = os.path.join(image_dir, f"{image_name}_segmentation_mask.png")
        Image.fromarray(result['mask']).save(mask_path)
        
        # 5. Optionally save depth mask if available
        if 'depth_mask' in result:
            depth_mask = (result['depth_mask'] * 255).astype(np.uint8)
            depth_mask_path = os.path.join(image_dir, f"{image_name}_depth_mask.png")
            Image.fromarray(depth_mask).save(depth_mask_path)
            
        print(f"Saved results for {result['filename']} to {image_dir}")
    
    print(f"All results saved to {base_output_dir}")
    return base_output_dir

def cluster_anomalies(anomalies, distance_threshold=50):
    """
    Cluster anomalies that are close to each other
    
    Args:
        anomalies: List of anomaly dictionaries
        distance_threshold: Maximum distance between anomalies to consider them as part of the same cluster
        
    Returns:
        clustered_anomalies: List of dictionaries, each containing a cluster of anomalies
    """
    if not anomalies:
        return []
    
    # Start with each anomaly in its own cluster
    clusters = [{
        'anomalies': [anomaly],
        'coords': anomaly['coords'],
        'contours': [anomaly.get('contour', None)],
        'patches': [anomaly['patch']]
    } for anomaly in anomalies]
    
    # Iteratively merge clusters
    merged = True
    while merged:
        merged = False
        new_clusters = []
        skip_indices = set()
        
        for i in range(len(clusters)):
            if i in skip_indices:
                continue
                
            cluster_i = clusters[i]
            merged_cluster = cluster_i.copy()
            
            for j in range(i + 1, len(clusters)):
                if j in skip_indices:
                    continue
                    
                cluster_j = clusters[j]
                x1, y1 = cluster_i['coords']
                x2, y2 = cluster_j['coords']
                
                # Calculate Euclidean distance
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance <= distance_threshold:
                    # Merge cluster_j into cluster_i
                    merged_cluster['anomalies'].extend(cluster_j['anomalies'])
                    merged_cluster['contours'].extend(cluster_j['contours'])
                    merged_cluster['patches'].extend(cluster_j['patches'])
                    
                    # Update centroid (average of all points)
                    all_coords = [a['coords'] for a in merged_cluster['anomalies']]
                    x_avg = sum(x for x, _ in all_coords) / len(all_coords)
                    y_avg = sum(y for _, y in all_coords) / len(all_coords)
                    merged_cluster['coords'] = (int(x_avg), int(y_avg))
                    
                    skip_indices.add(j)
                    merged = True
            
            new_clusters.append(merged_cluster)
        
        if merged:
            clusters = new_clusters
    
    # Format final clusters as new anomalies
    clustered_anomalies = []
    for i, cluster in enumerate(clusters):
        # Combine all patches into a larger one (optional)
        # For simplicity, just use the first patch
        representative_patch = cluster['patches'][0]
        representative_contour = next((c for c in cluster['contours'] if c is not None), None)
        
        clustered_anomaly = {
            'coords': cluster['coords'],
            'patch': representative_patch,
            'cluster_size': len(cluster['anomalies'])
        }
        
        if representative_contour is not None:
            clustered_anomaly['contour'] = representative_contour
            
        clustered_anomalies.append(clustered_anomaly)
    
    return clustered_anomalies

def depth_guided_anomaly_detection_with_clustering(image, depth_mask, threshold=55, patch_size=32, cluster_distance=50):
    """
    Perform anomaly detection using a depth mask and cluster nearby anomalies
    
    Args:
        image: RGB image as numpy array
        depth_mask: Boolean mask where True indicates pixels to include (close enough)
        threshold: Histogram threshold for anomaly detection
        patch_size: Size of patches to extract around anomalies
        cluster_distance: Distance threshold for clustering anomalies
        
    Returns:
        segmentation_mask: Binary mask of anomalous regions
        clustered_anomalies: List of anomalies with clustering applied
    """
    # First run standard depth-guided anomaly detection
    segmentation_mask, anomalies = depth_guided_anomaly_detection(
        image, depth_mask, threshold, patch_size
    )
    
    # Cluster the detected anomalies
    clustered_anomalies = cluster_anomalies(anomalies, cluster_distance)
    
    return segmentation_mask, clustered_anomalies



def anomaly_detection(image, threshold=55):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # Create segmentation mask
    segmentation_mask = np.ones_like(gray, dtype=np.uint8) * 255
    for intensity in range(256):
        if hist[intensity] > threshold:
            # This is a normal pixel intensity (occurs frequently)
            segmentation_mask[gray == intensity] = 0
    
    # Apply morphological operations to remove noise
    kernel = np.ones((10, 10), np.uint8)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    
    # Label connected components
    labeled_mask, num_features = ndimage.label(segmentation_mask)
    
    # Find all anomalies
    anomalies = []
    if num_features > 0:
        # For each connected component
        for i in range(1, num_features + 1):
            # Create mask for this component
            component_mask = (labeled_mask == i).astype(np.uint8) * 255
            
            # Find centroid
            M = cv2.moments(component_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Extract a patch around the anomaly
                patch_size = 100
                x1 = max(0, cx - patch_size//2)
                y1 = max(0, cy - patch_size//2)
                x2 = min(image.shape[1], cx + patch_size//2)
                y2 = min(image.shape[0], cy + patch_size//2)
                
                anomaly_patch = image[y1:y2, x1:x2]
                anomalies.append({
                    'coords': (cx, cy),
                    'patch': anomaly_patch,
                    'mask': component_mask
                })
    
    return segmentation_mask, anomalies

def visualize_results(image, segmentation_mask, anomalies):
    plt.figure(figsize=(14, 7))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Mark all anomalies
    for anomaly in anomalies:
        plt.plot(anomaly['coords'][0], anomaly['coords'][1], 'rx', markersize=10, linewidth=3)
    
    # Plot segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.title('Anomaly Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display each anomaly patch individually
    if anomalies:
        plt.figure(figsize=(15, 3 * len(anomalies)))
        for i, anomaly in enumerate(anomalies):
            plt.subplot(len(anomalies), 1, i+1)
            plt.imshow(anomaly['patch'])
            plt.title(f'Anomaly {i+1} at coordinates {anomaly["coords"]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def process_image(image_path, threshold=55):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Apply anomaly detection
    segmentation_mask, anomalies = anomaly_detection(image, threshold)
    
    # Visualize results
    visualize_results(image, segmentation_mask, anomalies)
    
    return segmentation_mask, anomalies

def depth_guided_anomaly_detection(image, depth_mask, threshold=55, patch_size=32):
    """
    Perform anomaly detection using a depth mask to ignore far pixels
    
    Args:
        image: RGB image as numpy array
        depth_mask: Boolean mask where True indicates pixels to include (close enough)
        threshold: Histogram threshold for anomaly detection
        patch_size: Size of patches to extract around anomalies
        
    Returns:
        segmentation_mask: Binary mask of anomalous regions
        anomalies: List of dicts with anomaly information
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert depth_mask to cv2-compatible format
    depth_mask_cv = depth_mask.astype(np.uint8) * 255
    
    # Calculate histogram only on pixels within depth threshold
    hist = cv2.calcHist([gray], [0], depth_mask_cv, [256], [0, 256])
    hist = hist.flatten()
    
    # Create segmentation mask - start with all zeros
    segmentation_mask = np.zeros_like(gray, dtype=np.uint8)
    
    # Mark infrequent pixels as anomalies (keeping original logic)
    for intensity in range(256):
        if hist[intensity] <= threshold:  # Infrequent pixel intensity
            # But only mark pixels that are within the depth range
            segmentation_mask[(gray == intensity) & depth_mask] = 255
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract information about each anomaly
    anomalies = []
    for contour in contours:
        # Find the center of the contour
        M = cv2.moments(contour)
        if M["m00"] > 0:  # Ensure the contour has area
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Extract a patch around the anomaly
            x1 = max(0, cx - patch_size // 2)
            y1 = max(0, cy - patch_size // 2)
            x2 = min(image.shape[1], cx + patch_size // 2)
            y2 = min(image.shape[0], cy + patch_size // 2)
            patch = image[y1:y2, x1:x2]
            
            # Only add if the patch isn't empty
            if patch.size > 0:
                anomalies.append({
                    'coords': (cx, cy),
                    'contour': contour,
                    'patch': patch
                })
    
    return segmentation_mask, anomalies

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "data/1735659675_edited_originalsize.png"
    image_path = "data/original (6).jpg"
    image_path = "data/original (5).jpg"
    
    # Process the image with different options
    # Process the image
    segmentation_mask, anomalies = process_image(image_path, threshold=550)
    
    print(f"Found {len(anomalies)} anomalous objects")
    
    # Enhanced algorithm (using RGB)
    # segmentation_mask2, coords2, patch2 = process_image(image_path, threshold=35, use_rgb=True, overlay=True)