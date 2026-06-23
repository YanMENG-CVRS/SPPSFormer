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
                        seed_normal, plane_point = update_plane_params(cluster_points)       

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

folder_path = 'out/out_rn3d'


new_folder_path = folder_path + '_RefinedBoundary'
os.makedirs(new_folder_path, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(new_folder_path, filename)

        print(f"正在处理文件: {filename}")

        data = np.loadtxt(file_path)   
        points = data[:, :3] 
        labels = data[:, 7].astype(int)  
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

        normals, curvature = estimate_normals_and_curvature(points)

        radius = 0.9
        nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
        adjacency_matrix = nbrs.radius_neighbors_graph(points).toarray()

        boundary_points = []
        neighbor_labels = {}

        for i in range(points.shape[0]):
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            neighbor_labels_set = set(labels[neighbors])
            neighbor_labels_set.discard(-1)  
            if len(neighbor_labels_set) > 1: 
                boundary_points.append(i)
                neighbor_labels[i] = neighbor_labels_set
                labels[i] = -1  

        boundary_points = np.array(boundary_points)

        plane_params = {}
        unique_labels = np.unique(labels[labels >= 0])  

        for label in unique_labels:
            plane_points = points[labels == label]
            if plane_points.shape[0] > 0:  
                centroid = plane_points.mean(axis=0)
                centered_points = plane_points - centroid
                _, _, Vt = np.linalg.svd(centered_points)
                normal = Vt[-1]
                if normal[2] < 0:
                    normal = -normal
                plane_params[label] = (centroid, normal)
        p2p_weight = 20

        for idx in boundary_points:
            min_distance = float("inf")
            best_label = -1
            for label in neighbor_labels[idx]: 
                if label not in plane_params:
                    continue  
                centroid, normal = plane_params[label]
                point_to_plane_dist = abs(np.dot(normal, points[idx] - centroid))
                normal_cosine_dist = 1 - np.dot(normals[idx], normal)
                combined_dist = p2p_weight * point_to_plane_dist + normal_cosine_dist
                if combined_dist < min_distance:
                    min_distance = combined_dist
                    best_label = label
            labels[idx] = best_label

        output_filename = os.path.splitext(new_file_path)[0] + "_RefinedBoundary.txt"

        data[:, 7] = labels
        np.savetxt(output_filename, data, fmt="%.6f %.6f %.6f %d %d %d %d %d")

        print(f"Segmentation result saved as: {output_filename}")
