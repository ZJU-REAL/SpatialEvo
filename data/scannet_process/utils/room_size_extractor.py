from turtle import down
import numpy as np
from scipy.spatial import Delaunay
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

def extract_floor_points(points, height_percentille=40, max_height_threshold=0.1):
    z_coords = points[:, 2]
    
    base_height = np.percentile(z_coords, height_percentille)
    
    floor_mask = z_coords < base_height + max_height_threshold
    floor_points = points[floor_mask]
    
    # print(f"Extracted {floor_points.shape[0]} floor points from {points.shape[0]} total points.")
    
    return floor_points

def down_sample_points(points, voxel_size=1):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        idx_tuple = tuple(idx)
        if idx_tuple not in voxel_dict:
            voxel_dict[idx_tuple] = i
    
    indices = list(voxel_dict.values())
    
    # print(f"Downsampled from {points.shape[0]} to {len(indices)} points.")
    
    return points[indices]
    

def calculate_room_area(points, alpha=1, visualize=False):
    floor_points = extract_floor_points(points)
    downsampled_points = down_sample_points(floor_points)
    points_2d = downsampled_points[:, :2].copy()
    
    np.random.seed(42)  
    noise = np.random.normal(0, 0.001, size=points_2d.shape) 
    points_2d_with_noise = points_2d + noise
    
    try:
        alpha_shape = alphashape.alphashape(points_2d_with_noise, alpha)
        
        if isinstance(alpha_shape, MultiPolygon):
            total_area = sum(polygon.area for polygon in alpha_shape.geoms)
        else:
            total_area = alpha_shape.area
            
        if visualize:
            plt.figure(figsize=(10, 10))
            
            plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, color='blue')
            
            if isinstance(alpha_shape, MultiPolygon):
                for polygon in alpha_shape.geoms:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y, 'r-')
            else:
                x, y = alpha_shape.exterior.xy
                plt.plot(x, y, 'r-')
                
            plt.axis('equal')
            plt.title(f'Room Area: {total_area:.2f} square meters')
            plt.savefig('room_area_visualization.jpg')
            plt.close()
            
        return total_area
        
    except Exception as e:
        print(f"Error calculating with Alpha Shape: {e}")
        
        try:
            hull = Delaunay(points_2d).convex_hull
            vertices = np.unique(hull)
            hull_points = points_2d[vertices]
            hull_polygon = Polygon(hull_points)
            return hull_polygon.area
        except Exception as e2:
            print(f"Error calculating with Convex Hull: {e2}")
            return 0.0
        
