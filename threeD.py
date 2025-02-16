import requests
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm   
import cv2
import os

def download_sample_dataset(object_name='dinosaur', output_dir='dataset'):
    """
    Download sample datasets for 3D reconstruction.
    Available objects: 'dinosaur'
    
    Args:
        object_name (str): Name of the object dataset to download
        output_dir (str): Directory to save the dataset
    """
    # Dictionary of sample datasets
    datasets = {
        'dinosaur': 'https://github.com/alicevision/dataset_monstree/archive/refs/heads/master.zip'
    }
    
    if object_name not in datasets:
        raise ValueError(f"Dataset '{object_name}' not found. Available datasets: {list(datasets.keys())}")
    
    url = datasets[object_name]
    output_path = Path(output_dir) / object_name
    zip_path = output_path.with_suffix('.zip')
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {object_name} dataset...")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as file, tqdm(
        desc=zip_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    # Extract dataset
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    # Clean up zip file
    zip_path.unlink()
    
    # Move images to root of output directory
    for img_file in output_path.rglob('*.jpg'):
        shutil.move(str(img_file), str(output_path / img_file.name))
    
    # Remove empty directories
    for dir_path in list(output_path.rglob('*'))[::-1]:
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()
    
    print(f"\nDataset downloaded to: {output_path}")
    print(f"Number of images: {len(list(output_path.glob('*.jpg')))}")
    return str(output_path)

def verify_dataset(dataset_path):
    """
    Verify downloaded dataset is suitable for 3D reconstruction.
    
    Args:
        dataset_path (str): Path to dataset directory
    """
    path = Path(dataset_path)
    images = list(path.glob('*.jpg'))
    
    print("\nDataset Verification:")
    print(f"Total images: {len(images)}")
    
    if len(images) < 20:
        print("⚠️  Warning: Less than 20 images found. More images recommended.")
    else:
        print("✓ Sufficient number of images found.")
    
    # Check image properties
    resolutions = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            height, width = img.shape[:2]
            resolutions.append((width, height))
    
    if resolutions:
        avg_width = sum(r[0] for r in resolutions) / len(resolutions)
        avg_height = sum(r[1] for r in resolutions) / len(resolutions)
        print(f"Average resolution: {int(avg_width)}x{int(avg_height)}")
        
        if avg_width < 2000 or avg_height < 2000:
            print("⚠️  Warning: Resolution might be low for good reconstruction.")
        else:
            print("✓ Image resolution is adequate.")

def reconstruct_3d_from_images(dataset_path):
    """
    Reconstruct 3D model from images using COLMAP.
    
    Args:
        dataset_path (str): Path to directory containing images
        
    Returns:
        tuple: (points3D, colors) where points3D is a numpy array of 3D points
              and colors is a numpy array of corresponding RGB colors
    """
    try:
        import numpy as np
        from pathlib import Path
        import subprocess
        import os
        
        dataset_path = Path(dataset_path)
        sparse_dir = dataset_path / 'sparse'
        dense_dir = dataset_path / 'dense'
        
        # Create necessary directories
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        # Run COLMAP commands
        print("\nRunning COLMAP reconstruction pipeline...")
        
        # Feature extraction
        database_path = os.path.abspath(str(dataset_path / 'database.db'))
        image_path = os.path.abspath(str(dataset_path))
        
        colmap_cmd = [
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', image_path,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
            '--SiftExtraction.max_num_features', '8192',
            '--SiftExtraction.max_image_size', '3200',
            '--SiftExtraction.num_threads', '4'
        ]
        
        try:
            result = subprocess.run(colmap_cmd, check=True, capture_output=True, text=True)
            print("COLMAP output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"COLMAP error output: {e.stderr}")
            raise
        
        # Feature matching
        subprocess.run([
            'colmap', 'exhaustive_matcher',
            '--database_path', str(dataset_path / 'database.db'),
            '--SiftMatching.use_gpu', '0'  # Force CPU matching
        ], check=True, shell=True)
        
        # Sparse reconstruction
        subprocess.run([
            'colmap', 'mapper',
            '--database_path', str(dataset_path / 'database.db'),
            '--image_path', str(dataset_path),
            '--output_path', str(sparse_dir)
        ], check=True,shell = True)
        
        # Dense reconstruction
        subprocess.run([
            'colmap', 'image_undistorter',
            '--image_path', str(dataset_path),
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(dense_dir),
            '--output_type', 'COLMAP'
        ], check=True,shell = True)
        
        subprocess.run([
            'colmap', 'patch_match_stereo',
            '--workspace_path', str(dense_dir)
        ], check=True,shell = True)
        
        subprocess.run([
            'colmap', 'stereo_fusion',
            '--workspace_path', str(dense_dir),
            '--output_path', str(dense_dir / 'fused.ply')
        ], check=True,shell = True)
        
        # Convert to mesh using Poisson surface reconstruction
        subprocess.run([
            'colmap', 'poisson_mesher',
            '--input_path', str(dense_dir / 'fused.ply'),
            '--output_path', str(dataset_path / 'mesh.ply')
        ], check=True,shell = True)
        
        print("\n3D reconstruction completed!")
        print(f"Mesh saved to: {dataset_path / 'mesh.ply'}")
        
        # Return the dense point cloud path for visualization
        return str(dense_dir / 'fused.ply')
        
    except subprocess.CalledProcessError as e:
        print(f"Error during COLMAP execution: {e}")
        return None
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return None

def save_point_cloud(points, colors, output_path):
    """Save point cloud to PLY file."""
    if points is None or colors is None:
        return
        
    with open(output_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points and colors
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    
    print(f"\nPoint cloud saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    try:
        # Download dinosaur dataset
        dataset_path = download_sample_dataset('dinosaur')
        
        # Verify the dataset
        verify_dataset(dataset_path)
        
        # Reconstruct 3D model
        point_cloud_path = reconstruct_3d_from_images(dataset_path)
        
        if point_cloud_path:
            # Visualize the result
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            o3d.visualization.draw_geometries([pcd])
            
            print("\nYou can now import the mesh.ply file into Blender!")
            print("Steps in Blender:")
            print("1. File -> Import -> Stanford (.ply)")
            print("2. Navigate to your dataset folder")
            print("3. Select mesh.ply and import")
    
    except Exception as e:
        print(f"Error: {e}")