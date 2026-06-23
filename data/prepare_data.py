import glob
import multiprocessing as mp
import numpy as np
import open3d as o3d
import torch
from spformer.utils import boundary_detection_xyz

remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def prepare_data_train(fn):
    with open(fn, "r") as f:
        lines_as_lists = []
        for line in f:
            columns = line.strip().split()
            lines_as_lists.append(columns)

        points = np.ascontiguousarray(np.array(lines_as_lists, dtype=np.float32))
        coords = np.ascontiguousarray(points[:, 0:3]-points[:, 0:3].mean(0))
        colors =np.zeros((points.shape[0], 3))
        fourth_row = points[:, 3]
        sem_labels = np.where(fourth_row > 0, 1, 0).astype(np.float64)
        sem_labels = np.ascontiguousarray(sem_labels.astype(np.float64))
        seg = np.ascontiguousarray(points[:, 3].astype(np.float64))
        segid_to_pointid = {}
        for i in range(len(points)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)

        instance_segids = []
        labels = []
        instance_labels = seg
        fourth_row = points[:, 4]

        superpoint = (fourth_row - 1).astype(np.int64)
        boundary = boundary_detection_xyz(coords, show=False)
        torch.save((coords, colors, superpoint, sem_labels, instance_labels, boundary), fn[:-4]+'_train3.pth')

from tqdm import tqdm
import os
if __name__ == "__main__":
    split = 'train'
    files = sorted(glob.glob(split+'/*.txt'))
    p = mp.Pool(processes=mp.cpu_count())
    for f in tqdm(files, desc="Processing files"):
        prepare_data_test(f)
