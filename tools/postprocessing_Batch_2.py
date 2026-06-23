from symbol import continue_stmt

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree
import pyvista as pv
from sklearn.preprocessing import normalize
import os
import math


def estimate_normals_and_curvature(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)
    curvature = np.zeros(points.shape[0])

    for i, neighbors in enumerate(indices):
        neighbors = neighbors[1:]  
        p_neighbors = points[neighbors] - points[i]
        cov_matrix = np.cov(p_neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        normals[i] = eigenvectors[:, 0]
        if normals[i][2] < 0:
            normals[i] = -normals[i]
        curvature[i] = eigenvalues[0] / np.sum(eigenvalues)

    return normals, curvature

def region_growing(points, normals, seed_indices, noise_is_isolated, start_label = 3, angle_threshold=np.pi / 3, distance_threshold=0.5, radius=0.9, min_cluster_size=20, update_interval=10):
    labels = -np.ones(points.shape[0], dtype=int)        
    current_label = start_label

    kdtree = cKDTree(points)       

    def update_plane_params(cluster_points):
        centroid = np.mean(cluster_points, axis=0)        
        cov_matrix = np.cov(cluster_points - centroid, rowvar=False)        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)       
        normal = eigenvectors[:, np.argmin(eigenvalues)]        
        if normal[2] < 0:
           normal = -normal
        return normal, centroid

    for seed_index in seed_indices:        
        if labels[seed_index] != -1:        
            continue

        seed_queue = [seed_index]       
        current_cluster = []            
        labels[seed_index] = current_label        
        current_cluster.append(seed_index)        
        seed_normal = normals[seed_index]        
        if seed_normal[2] < 0:
           seed_normal = -seed_normal
        plane_point = points[seed_index]         

        while seed_queue:
            current_point_index = seed_queue.pop(0)       
            current_point_normal = normals[current_point_index]        

            neighbors = kdtree.query_ball_point(points[current_point_index], r=radius)        

            for neighbor in neighbors:        
                if labels[neighbor] != -1:        
                    continue

                neighbor_point = points[neighbor]        
                neighbor_normal = normals[neighbor]        

                normal_angle = np.arccos(np.clip(np.dot(seed_normal, neighbor_normal), -1.0, 1.0))        
                point_to_plane_distance = np.abs(np.dot(seed_normal, (neighbor_point - plane_point)))     

                if normal_angle < angle_threshold and point_to_plane_distance < distance_threshold:     
                    labels[neighbor] = current_label      
                    seed_queue.append(neighbor)       
                    current_cluster.append(neighbor)       

                    if len(current_cluster) % update_interval == 0:        
                        cluster_points = points[current_cluster]        
                        seed_normal, plane_point = update_plane_params(cluster_points)        #？？？？

        if len(current_cluster) < min_cluster_size:       
            for idx in current_cluster:     
                labels[idx] = -1       
        else:
            isolation_ratio = np.mean(noise_is_isolated[current_cluster])
            if isolation_ratio > 0.5:
                labels[current_cluster] = -1 
            else:
                current_label += 1  

    return labels

def visualize_segmentation(points, labels, UItitle="Unamed UI"):
    point_cloud = pv.PolyData(points)
    point_cloud['labels'] = labels

    label_colors = {
        -1: 'blue',
        0: 'red',
        1: 'green',
        2: 'yellow',
        3: 'purple',
        4: 'orange',
        5: 'cyan',
        6: 'magenta',
        7: 'brown',
        8: 'pink',
        9: 'lime',
        10: 'olive',
        11: 'navy',
        12: 'teal',
        13: 'coral',
        14: 'gold',
        15: 'maroon',
        16: 'turquoise',
        17: 'violet',
        18: 'indigo',
        19: 'khaki',
    }

    plotter = pv.Plotter()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        color = label_colors.get(label, 'gray')
        plotter.add_points(point_cloud.extract_points(labels == label), color=color)

    plotter.show(title=UItitle)

folder_path = r'out/out_rn3d'


new_folder_path = folder_path + '_PostSeg_2'
os.makedirs(new_folder_path, exist_ok=True)

for filename in os.listdir(folder_path):

    if filename.endswith('.txt'):

        file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(new_folder_path, filename)

        print(f"正在处理文件: {filename}")

        data = np.loadtxt(file_path)
        xyz = data[:, :3]  # XYZ
        labels = data[:, 7].astype(int) 

        unique_labels, counts = np.unique(labels, return_counts=True)
        small_planes = unique_labels[counts < 20]  
        for label in small_planes:
            labels[labels == label] = -1  

        normals, curvature = estimate_normals_and_curvature(xyz)

        k = 10
        tree = cKDTree(xyz)
        distances, _ = tree.query(xyz, k=k + 1)  
        avg_distances = np.mean(distances[:, 1:], axis=1)  

        valid_labels = labels != -1 
        distance_threshold = np.percentile(avg_distances[valid_labels], 99) 
        z_threshold_top = xyz[valid_labels, 2].max()  
        z_threshold_bottom = xyz[valid_labels, 2].min()

        is_isolated = (avg_distances > distance_threshold) | (xyz[:, 2] < z_threshold_bottom - 0.2) | (
                    xyz[:, 2] > z_threshold_top + 0.2)  

        sparse_labels = np.where((avg_distances <= distance_threshold) & (xyz[:, 2] >= z_threshold_bottom - 0.2) & (
                    xyz[:, 2] <= z_threshold_top + 0.2), 0, -1)

        cluster_points = data[labels != -1]
        cluster_normals = normals[labels != -1]
        plane_params = []

        distance_stats = []

        for label in np.unique(cluster_points[:, 7]):
            points = cluster_points[cluster_points[:, 7] == label][:, :3]
            centroid = points.mean(axis=0)
            points_centered = points - centroid

            ransac = RANSACRegressor()
            ransac.fit(points_centered[:, :2], points_centered[:, 2])
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            normal = np.array([-coef[0], -coef[1], 1.0])
            normal = normalize(normal[:, np.newaxis], axis=0).ravel()


            distances = np.abs(np.dot(points_centered, normal) - intercept) / np.linalg.norm(normal)
            normals_plane = cluster_normals[cluster_points[:, 7] == label]
            normal_diffs = 1 - np.einsum('ij,ij->i', normals_plane, normal[np.newaxis, :])

            distance_stats.append({
                'max_distance': distances.max(),
                'max_cosine_dist': normal_diffs.max()
            })

        distance_threshold = max(stat['max_distance'] for stat in distance_stats)
        if distance_threshold < 0.5:
            distance_threshold = 0.5
        angle_threshold = max(stat['max_cosine_dist'] for stat in distance_stats)
        angle_threshold = 1 - angle_threshold
        if angle_threshold < 0.5:  
            angle_threshold = math.acos(0.5)
        else:
            angle_threshold = math.acos(angle_threshold)

        noise_points = xyz[labels == -1]
        noise_normals = normals[labels == -1]
        noise_curvature = curvature[labels == -1]
        noise_is_isolated = is_isolated[labels == -1]  

        seed_indices = np.argsort(noise_curvature)[:int(0.5 * len(noise_curvature))]
        seed_points = noise_points[seed_indices]
        seed_normals = noise_normals[seed_indices]
        start_label = labels.max() + 1
        if len(noise_points)>0:

            new_labels = region_growing(noise_points, noise_normals, seed_indices, noise_is_isolated, start_label,
                angle_threshold, distance_threshold)
        else:
            new_labels = []

        data[labels == -1, 7] = new_labels

        output_filename = os.path.splitext(new_file_path)[0] + "_Seg.txt"
        fmt = ["%.10f", "%.10f", "%.10f"] + ["%d"] * (data.shape[1] - 3)
        np.savetxt(output_filename, data, fmt=fmt)

        print(f"Segmentation result saved as: {output_filename}")

