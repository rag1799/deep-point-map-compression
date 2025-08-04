
import os
import numpy as np
import argparse
from numpy.linalg import inv
import time
from ruamel.yaml import YAML
from sklearn.cluster import MiniBatchKMeans
import open3d as o3d
import depoco.utils.point_cloud_utils as pcu
import octree_handler

# === SEMANTIC LABEL MAPPING ===
ORIGINAL_IDS = np.array([
    10, 252, 15, 255, 18, 258, 20, 259,
    30, 254, 31, 253, 40, 44, 48, 49,
    50, 51, 70, 71, 72
], dtype=np.uint16)

NEW_IDS = np.array([
    1, 1, 2, 2, 3, 3, 4, 4,
    5, 5, 6, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15
], dtype=np.uint8)

UNWANTED_CLASSES = np.array([0, 1, 52, 60, 99], dtype=np.uint16)

NUM_CLASSES = NEW_IDS.max() + 1

def label_to_one_hot(labels):
    one_hot = np.zeros((labels.shape[0], NUM_CLASSES), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def open_label(filename):
    label = np.fromfile(filename, dtype=np.uint32) & 0xFFFF
    return label

def parse_calibration(filename):
    calib = {}
    with open(filename) as f:
        for line in f:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            if len(values) == 12:
                pose = np.eye(4)
                pose[:3] = np.array(values).reshape(3, 4)
                calib[key] = pose
    return calib

def parse_poses(filename, calibration):
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)
    with open(filename) as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[:3] = np.array(values).reshape(3, 4)
            poses.append(Tr_inv @ pose @ Tr)
    return poses

def group_points(points, labels, group_size=5):
    if len(points) == 0:
        return np.empty((0, 3 + 1 + NUM_CLASSES), dtype=np.float32)

    results = []
    for lbl in np.unique(labels):
        mask = labels == lbl
        class_points = points[mask]

        if len(class_points) <= group_size:
            one_hot = label_to_one_hot(np.full((1,), lbl))
            avg = np.mean(class_points, axis=0)
            results.append(np.concatenate([avg[:3], [avg[3]], one_hot[0]]))
            continue

        n_clusters = max(1, len(class_points) // group_size)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init_size=3*n_clusters)
        kmeans.fit(class_points[:, :3])
        clusters = kmeans.predict(class_points[:, :3])

        for c in range(n_clusters):
            c_mask = clusters == c
            if not np.any(c_mask):
                continue
            c_pts = class_points[c_mask]
            avg = np.mean(c_pts, axis=0)
            one_hot = label_to_one_hot(np.full((1,), lbl))
            results.append(np.concatenate([avg[:3], [avg[3]], one_hot[0]]))

    return np.array(results, dtype=np.float32)

class Kitti2voxelModified:
    def __init__(self, config):
        self.config = config
        self.folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(seq) 
                        for seq in config["dataset"]["data_folders"]["train"]]

    def process_sequence(self):
        for seq_path in self.folders:
            out_dir = pcu.path(self.config["dataset"]["data_folders"]["grid_output"]) + pcu.path(seq_path.split("/")[-2])
            os.makedirs(out_dir, exist_ok=True)
            calib = parse_calibration(seq_path + "calib.txt")
            poses = parse_poses(seq_path + "poses.txt", calib)
            scan_dir = os.path.join(seq_path, "velodyne")
            files = sorted([f for f in os.listdir(scan_dir) if f.endswith(".bin")])
            for fname in files:
                scan = np.fromfile(os.path.join(scan_dir, fname), dtype=np.float32).reshape(-1, 4)
                label_file = os.path.join(seq_path, "labels", fname.replace(".bin", ".label"))
                if not os.path.exists(label_file): continue
                raw_labels = open_label(label_file)

                # Apply filters
                valid_mask = ~np.isin(raw_labels, UNWANTED_CLASSES)
                scan = scan[valid_mask]
                raw_labels = raw_labels[valid_mask]

                # Remap labels
                label_map = np.zeros_like(raw_labels, dtype=np.uint8)
                for orig, new in zip(ORIGINAL_IDS, NEW_IDS):
                    label_map[raw_labels == orig] = new

                # Group points
                grouped = group_points(scan, label_map, group_size=self.config["grid"]["group_size"])
                if len(grouped) == 0:
                    continue

                # Estimate normals
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(grouped[:, :3])
                if len(grouped) >= 3:
                    try:
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=self.config['grid']['normal_eigenvalue_radius'], max_nn=30))
                        pcd.orient_normals_to_align_with_direction()
                        normals = np.asarray(pcd.normals)
                    except:
                        normals = np.zeros((len(grouped), 3))
                else:
                    normals = np.zeros((len(grouped), 3))

                # Save
                final = np.hstack([grouped, normals])
                out_path = os.path.join(out_dir, fname)
                final.astype(np.float32).tofile(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_cfg", "-cfg", type=str, required=True)
    args = parser.parse_args()
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(open(args.arch_cfg, 'r'))
    processor = Kitti2voxelModified(config)
    processor.process_sequence()
