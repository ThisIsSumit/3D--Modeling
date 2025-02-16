import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog

def view_point_cloud(file_path=None):
    """
    Interactive point cloud viewer with controls
    
    Controls:
    - Left click + drag: Rotate
    - Right click + drag: Pan
    - Mouse wheel: Zoom
    - 'H': Show help menu
    - 'R': Reset view
    - '-'/'+': Decrease/Increase point size
    """
    
    if file_path is None:
        # Create file dialog to select .ply file
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if not file_path:
            print("No file selected")
            return
    
    # Load the point cloud
    print(f"Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Create visualization settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud Viewer", width=1024, height=768)
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Get render options and set initial settings
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 1.0
    
    # Get view control
    view_control = vis.get_view_control()
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5,  # Size of the coordinate frame
        origin=[0, 0, 0]  # Position of the coordinate frame
    )
    vis.add_geometry(coordinate_frame)
    
    print("\nControls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("- 'H': Show help menu")
    print("- 'R': Reset view")
    print("- '-'/'+': Decrease/Increase point size")
    print("- 'Q' or 'Esc': Quit")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def create_sample_point_cloud():
    """Create a sample point cloud if no file is available"""
    # Create a simple cube point cloud
    points = []
    colors = []
    
    # Create points for a cube
    for x in np.linspace(-1, 1, 20):
        for y in np.linspace(-1, 1, 20):
            for z in np.linspace(-1, 1, 20):
                points.append([x, y, z])
                # Add some color variation based on position
                colors.append([
                    (x + 1) / 2,  # Red varies with x
                    (y + 1) / 2,  # Green varies with y
                    (z + 1) / 2   # Blue varies with z
                ])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save the sample point cloud
    output_file = "sample_cube.ply"
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Created sample point cloud: {output_file}")
    return output_file

if __name__ == "__main__":
    try:
        # Try to load and view a point cloud file
        view_point_cloud( 'dataset/dinosaur/images/00000000.jpg.ply' )
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        print("Creating and viewing a sample point cloud instead...")
        sample_file = create_sample_point_cloud()
        view_point_cloud(sample_file)