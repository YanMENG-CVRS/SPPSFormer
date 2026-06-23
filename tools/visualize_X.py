import numpy as np
import os
import os.path as osp
import torch
import re
import glob
from tqdm import tqdm

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255

def calculate_iou(predict_set, true_set):
    indices = np.where(predict_set)[0]
    indices2 = np.where(true_set)[0]
    intersection = len(np.intersect1d(indices, indices2))
    union = len(np.union1d(indices, indices2))
    iou = intersection / union if union != 0 else 0
    return iou

def get_coords_color(room_name, data_path, out_path):
    file = osp.join(data_path, room_name + '_train3.pth')
    file2 = osp.join(data_path, room_name + '.txt')
    with open(file2, "r") as f:
        lines = [line.strip().split() for line in f]
    points = np.ascontiguousarray(np.array(lines, dtype=np.float64))
    xyz = np.ascontiguousarray(points[:, 0:3])
    inst_label = np.ascontiguousarray(points[:, 3]).astype(int)

    xyz2, rgb, superpoint, label, inst_label2, boundary = torch.load(file)
    rgb = (rgb + 1) * 127.5

    instance_file = os.path.join(out_path, 'pred_instance', room_name + '.txt')
    assert os.path.isfile(instance_file), f'No instance result - {instance_file}.'
    with open(instance_file, 'r') as f:
        masks = [line.rstrip().split() for line in f]

    predict_label_pred_rgb = np.zeros(rgb.shape)
    ins_pointnum = np.zeros(len(masks))
    predict_label = -100 * np.ones(rgb.shape[0], dtype=int)

    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]

    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        mask_path = os.path.join(out_path, 'pred_instance', masks[i][0])
        if float(masks[i][2]) < 0.06:
            continue
        mask = np.array(open(mask_path).read().splitlines(), dtype=int)
        ins_pointnum[i] = mask.sum()
        predict_label[mask == 1] = i

    sort_idx = np.argsort(ins_pointnum)[::-1]
    for i in range(len(masks)):
        predict_label_pred_rgb[predict_label == sort_idx[i]] = COLOR_DETECTRON2[i % len(COLOR_DETECTRON2)]
    rgb = predict_label_pred_rgb.astype(np.int32)

    iou_values = []
    for true_label_id in np.unique(inst_label):
        if true_label_id != 0:
            true_mask = (inst_label == true_label_id)
            max_iou = 0
            for predict_label_id in np.unique(predict_label):
                if predict_label_id != -100:
                    predict_mask = (predict_label == predict_label_id)
                    iou = calculate_iou(predict_mask, true_mask)
                    max_iou = max(max_iou, iou)
            iou_values.append(max_iou)

    miou = np.mean(iou_values)
    print(f"mIoU: {miou:.4f}")
    return xyz, rgb, inst_label, predict_label

def map_different(x):
    unique_positive_values = np.unique(x[x >= 0])
    value_to_index = {val: idx for idx, val in enumerate(unique_positive_values, start=0)}
    return np.array([value_to_index[val] if val in value_to_index else -1 for val in x])

def process_rn3d_predictions(room_names_folder, inst_out_path, out_path, file_pattern=r'building2048_(\d+)'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    txt_files = glob.glob(os.path.join(room_names_folder, 'updated_building2048_*_X10.txt'))

    for file_path in tqdm(txt_files, desc='Processing files'):
        match = re.search(file_pattern, os.path.basename(file_path))
        if match:
            extracted_string = 'updated_building2048_' + match.group(1) + '_X10'
            print(f"Processing: {extracted_string}")
        else:
            continue

        room_name = extracted_string
        output_file = osp.join(out_path, room_name + '.txt')
        data_path = room_names_folder
        xyz, rgb, inst_label, predict_label = get_coords_color(room_name, data_path, inst_out_path)
        predict_label = map_different(predict_label)

        with open(output_file, 'w') as f_combined:
            for point, color, inst, predict in zip(xyz, rgb, inst_label, predict_label):
                f_combined.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {inst} {predict}\n")

if __name__ == '__main__':
    process_rn3d_predictions(
        room_names_folder='data/rn3d_patch_kmeans/val',
        inst_out_path='SPPSFormer/out_rn3d',
        out_path='SPPSFormer/out/out_rn3d',
        file_pattern=r'updated_building2048_(\d+)_X10' 
    )

