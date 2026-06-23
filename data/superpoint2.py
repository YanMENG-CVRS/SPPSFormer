import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

def normalize_coordinates(coord_str_list):
    coord_str = ' '.join(coord_str_list)
    parts = coord_str.split()
    return tuple(float(part) for part in parts)

def read_color_data(file_path):
    color_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            coord_key = normalize_coordinates(parts[:3])
            color_data = parts[3:]
            color_map[coord_key] = color_data
    return color_map

def assign_instance_ids(points_list):
    color_to_instance = {(0, 0, 0): 0}  
    instance_counter = 1  

    points_list_1 = []
    for i, point in enumerate(points_list):
        parts = []
        parts = point
        color = tuple(map(int, parts[-3:]))  

        if color not in color_to_instance:
            color_to_instance[color] = instance_counter
            instance_counter += 1
        instance_id = color_to_instance[color]
        parts[-3:] = [str(instance_id)]  
        float_data = [float(item) for item in parts]
        float_int_data = [float(float_data[0]),float(float_data[1]),float(float_data[2]),  int(float_data[-2]), int(float_data[-1])]
        points_list_1.append(float_int_data)
    return points_list_1, color_to_instance


def merge_data(file_path1, color_map):
    merged_lines = []
    with open(file_path1, 'r') as file1:
        for line in file1:
            parts = line.strip().split()
            coord_key = normalize_coordinates(parts[:3])
            color_info = color_map.get(coord_key, ['0', '0', '0'])  
            merged_line = f"{' '.join(parts)} {' '.join(color_info)}"
            merged_line = merged_line.split()
            merged_lines.append(merged_line)
        extracted_lines = [] 
        for line in merged_lines:
            if line[-3:] == ['0', '0', '0']:
                extracted_lines.append(line)  
    merged_lines_copy = copy.deepcopy(merged_lines)
    merged_lines2,colors = assign_instance_ids(merged_lines_copy)
    merged_data3 = patch_more(merged_lines2)
    extracted_lines2,colors2 = assign_instance_ids(extracted_lines)
    return merged_data3,extracted_lines2

import torch
def patch_more(data):
    from collections import defaultdict
    from sklearn.cluster import KMeans
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
    patches = defaultdict(list)
    pcd = o3d.geometry.PointCloud()
    for entry in data:
        instance = entry[-1] 
        patches[instance].append(entry)  
    print(len(patches))
    max_labels = 0
    patch_with_label_list = []
    for i in range(1,len(patches)):
        pcd = o3d.geometry.PointCloud()
        points = [[x for x in line[:3]] for line in patches[i]]
        pcd.points = o3d.utility.Vector3dVector(points)
        k = 10
        keams = KMeans(n_clusters=k,n_init=10)
        if len(points)>=10:
            keams.fit(points)
            labels = keams.labels_

        else:
            keams2 = KMeans(n_clusters=len(points)//3, n_init=10)
            keams2.fit(points)
            labels = keams2.labels_
        patch_with_label = list(zip(patches[i],labels))
        for point, label in patch_with_label:
            new_point = [point[0], point[1], point[2], point[3], max_labels + label + 1]
            patch_with_label_list.append(new_point)
        max_labels = max_labels + max(labels)+1
    for point in patches[0]:
        new_point = [point[0], point[1], point[2], point[3], 0]
        patch_with_label_list.append(new_point)
    return patch_with_label_list

from collections import defaultdict
import torch

def patch_more_GPU(data):
    patches = defaultdict(list)
    pcd = o3d.geometry.PointCloud()
    for entry in data:
        instance = entry[-1] 
        patches[instance].append(entry)  
    print(len(patches))
    max_labels = 0
    patch_with_label_list = []
    for i in range(1,len(patches)):
        pcd = o3d.geometry.PointCloud()
        points = [[x for x in line[:3]] for line in patches[i]]
        pcd.points = o3d.utility.Vector3dVector(points)
        k = 10
        if len(points)>=10:
            x = torch.tensor(points)
            cluster_ids_x, cluster_centers = kmeans(
                X=x, num_clusters=k, distance='euclidean', device=torch.device('cuda:0'), tqdm_flag=None
            )

            labels = cluster_ids_x

        else:
            x = torch.tensor(points)
            cluster_ids_x, cluster_centers = kmeans(
                X=x, num_clusters=len(points)//3, distance='euclidean', device=torch.device('cuda:0'), tqdm_flag=None
            )

            labels = cluster_ids_x

        labels = labels.tolist()
        patch_with_label = list(zip(patches[i],labels))
        for point, label in patch_with_label:
            new_point = [point[0], point[1], point[2], point[3], max_labels + label + 1]
            patch_with_label_list.append(new_point)
        max_labels = max_labels + max(labels)+1
    for point in patches[0]:
        new_point = [point[0], point[1], point[2], point[3], 0]
        patch_with_label_list.append(new_point)
    return patch_with_label_list

def write_to_data(update_path,merged_data):
    with open(update_path, 'w') as outfile:
        for point in merged_data:
            point_str = ' '.join(map(str, point)) + '\n'
            outfile.write(point_str)

def DBSCAN(data):
    pcd = o3d.geometry.PointCloud()

    points = [[float(x) for x in line[:3]] for line in data]
    pcd.points = o3d.utility.Vector3dVector(points)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=2, min_points=4, print_progress=True))
    max_label = max(labels)  
    assert len(data) == len(labels), "Points and labels must have the same length."
    print(f"all noise data number:{len(data)}")
    all_data = [[float(x) for x in line[:4]] for line in data]
    points_with_labels = list(zip(all_data, labels))
    points_with_labels_list = [
        [point[0], point[1], point[2],point[3], label] for point, label in points_with_labels]
    return points_with_labels_list

def update_marged_with_noise(merged_data, noise_patch_data):
    merged_data_last = [item[-1] for item in merged_data]
    max_value = max(merged_data_last)
    for noise_point in noise_patch_data:
        index_key = tuple(noise_point[:3])
        for i, merged_point in enumerate(merged_data):
            if tuple(merged_point[:3]) == index_key:
                a = merged_data[i]
                if noise_point[-1] == -1:
                    merged_data[i][-1] = 0
                else:
                    merged_data[i][-1] =max_value + noise_point[-1]+1

                b = merged_data[i]
                break

from tqdm import tqdm

import os
if __name__ == '__main__':
    folder_path = '/data/rn3d/train_origin'  
    folder_path_2 = 'data/rn3d/train_stage1'
    folder_path_out = '/data/rn3d/train_stage2'
    folder_path_noise_out = '/data/rn3d/train_noise'   
    if not os.path.exists(folder_path_out):

        os.makedirs(folder_path_out)
        print(f"Directory {folder_path_out} was created.")
    else:
        print(f"Directory {folder_path_out} already exists.")
    if not os.path.exists(folder_path_noise_out):
        os.makedirs(folder_path_noise_out)
        print(f"Directory {folder_path_noise_out} was created.")
    else:
        print(f"Directory {folder_path_noise_out} already exists.")



    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist.")
    else:
        for filename in tqdm(os.listdir(folder_path), desc='Processing files'):
            if filename.endswith('.txt'):
                file_path1 = os.path.join(folder_path, filename)
                file_path2 = os.path.join(folder_path_2, filename.replace('.txt', '_out.txt'))
                if not os.path.exists(file_path2):
                    print(f"The corresponding output file {file_path2} does not exist. Skipping {filename}.")
                    continue
                color_map = read_color_data(file_path2)
                merged_data, noise_data = merge_data(file_path1, color_map)
                update_path = os.path.join(folder_path_out, 'updated_' + filename)
                noise_path = os.path.join(folder_path_noise_out, 'noise_' + filename)
                print(f"DBSCAN {filename} .")
                if noise_data:
                    noise_data_patch = DBSCAN(noise_data)
                else:
                    noise_data_patch = []
                update_marged_with_noise(merged_data,noise_data_patch)
                write_to_data(update_path, merged_data)
                print(f"Processed and updated {filename} successfully.")
