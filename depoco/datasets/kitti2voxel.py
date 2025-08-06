import numpy as np
import argparse
from numpy.linalg import inv
import time
import os
from ruamel import yaml
from matplotlib import pyplot as plt
import open3d as o3d
import depoco.utils.point_cloud_utils as pcu
from pathlib import Path
import octree_handler
from collections import defaultdict, Counter
from ruamel.yaml import YAML 
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Define the label mapping and one-hot encoding parameters
LABEL_MAPPING = {
    1:  0,   # "unlabeled" (ignore)
    10: 1,   # "car"
    11: 2,   # "bicycle"
    13: 3,   # "bus"
    15: 4,   # "motorcycle"
    30: 5,   # "person"
    40: 6,   # "road"
    48: 7,   # "sidewalk"
    50: 8,   # "building"
    70: 9,   # "vegetation"
    # Ignore all other classes (map to 0 or a background class)
}

NUM_CLASSES = len(LABEL_MAPPING)

def label_to_one_hot(labels):
    """Convert labels to one-hot encoding"""
    # First map the labels to our reduced set of classes
    mapped_labels = np.zeros_like(labels, dtype=np.int32)
    for original_label, mapped_label in LABEL_MAPPING.items():
        mapped_labels[labels == original_label] = mapped_label
    
    # Create one-hot encoding
    valid_indices = (mapped_labels >= 0) & (mapped_labels < NUM_CLASSES)
    one_hot = np.zeros((labels.shape[0], NUM_CLASSES), dtype=np.int32)
    one_hot[np.arange(labels.shape[0])[valid_indices], mapped_labels[valid_indices]] = 1
    #one_hot = np.zeros((labels.shape[0], NUM_CLASSES), dtype=np.float32)
    #one_hot[np.arange(labels.shape[0]), mapped_labels] = 1
    
    return one_hot

def open_label(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    # if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
    #     raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF

    # set it
    return label

def parse_calibration(filename):
    """ read calibration file with given filename
        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        # print(key,values)
        if len(values) == 12:
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

    calib_file.close()
    print('calibration', calib)
    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def distanceMatrix(x, y):
    """ x[nxd], y[mxd],  d = dimensionality (eg. 3)
     distance between each point of x to each point of y
    Return
    -----
    distance matrix [nxm]

    """
    dims = x.shape[1]
    dist = np.zeros((x.shape[0], y.shape[0]))
    for i in range(dims):
        dist += (x[:, i][..., np.newaxis] - y[:, i][np.newaxis, ...])**2
    return dist**0.5


def getKeyPoses(pose_list, delta=50):
    '''
    creates a key pose for every pose which is delta apart (horizontal distance)
    returns
    -------
      idx, poses, distance_matrix [nxn]
    '''
    poses = np.asarray(pose_list)
    xy = poses[:, 0:2, -1]
    dist = distanceMatrix(xy, xy)

    key_pose_idx = []
    indices = np.arange(poses.shape[0])

    dist_it = dist.copy()
    while (dist_it.shape[0] > 0):
        key_pose_idx.append(indices[0])
        valid_idx = dist_it[0, :] > delta
        dist_it = dist_it[valid_idx, :]
        dist_it = dist_it[:, valid_idx]
        indices = indices[valid_idx]
    return key_pose_idx, poses[key_pose_idx], dist


class Kitti2voxelConverter():
    def __init__(self, config):
        self.config = config
        self.train_folders = []
        self.valid_folders = []
        self.test_folders = []
        if type(config["dataset"]["data_folders"]["train"]) is list:
            self.train_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["train"]]
        if type(config["dataset"]["data_folders"]["valid"]) is list:
            self.valid_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["valid"]]
        if type(config["dataset"]["data_folders"]["test"]) is list:
            self.test_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["test"]]

    def _process_scan(self, args):
        """Parallelized scan processing"""
        i, poses, seq_path, lower_bound, upper_bound = args
        sfile = seq_path + "velodyne/" + str(i).zfill(6)+'.bin'
        scan = np.fromfile(sfile, dtype=np.float32) if os.path.isfile(sfile) else np.zeros((0, 4))
        scan = scan.reshape((-1, 4))
        
        # Filter by distance
        dists = np.linalg.norm(scan[:, 0:3], axis=1)
        valid_p = (dists > self.config['grid']['min_range']) & (dists < self.config['grid']['max_range'])
        scan = scan[valid_p]
        
        # Transform to world coordinates
        scan_hom = np.ones((scan.shape[0], 4))
        scan_hom[:, 0:3] = scan[:, 0:3]
        points = np.matmul(poses[i], scan_hom.T).T[:, 0:3]
        
        # Load labels
        label = np.full((points.shape[0],), 2)
        if os.path.isfile(seq_path + "labels/" + str(i).zfill(6)+'.label'):
            label = open_label(filename=seq_path + "labels/" + str(i).zfill(6)+'.label')[valid_p]
        
        # Filter points
        valids = (np.all(points > lower_bound, axis=1) & 
                 np.all(points < upper_bound, axis=1) & 
                 (label < 200) & (label > 1))
        
        return points[valids], scan[valids, 3], label[valids]


    def sparsifieO3d(self, poses, key_pose_idx, seq_path, distance_matrix):
       
        grid_size = np.array((self.config['grid']['size']))
        center = poses[key_pose_idx][0:3, -1] + \
            np.array((0, 0, self.config['grid']['dz']))
        upper_bound = center + grid_size/2
        lower_bound = center - grid_size/2
        
        valid_scans = np.argwhere(
            distance_matrix[key_pose_idx, :] < grid_size[0] + self.config['grid']['max_range']).squeeze()
        
        # Collect all points, intensities, and raw labels
        all_points = []
        all_intensities = []
        all_labels = []
        
        for i in valid_scans:
            sfile = seq_path + "velodyne/" + str(i).zfill(6)+'.bin'
            scan = np.fromfile(sfile, dtype=np.float32) if os.path.isfile(sfile) else np.zeros((0, 4))
            scan = scan.reshape((-1, 4))
            
            # Filter by distance
            dists = np.linalg.norm(scan[:, 0:3], axis=1)
            valid_p = (dists > self.config['grid']['min_range']) & (dists < self.config['grid']['max_range'])
            scan = scan[valid_p]
            
            # Transform to world coordinates
            scan_hom = np.ones((scan.shape[0], 4))
            scan_hom[:, 0:3] = scan[:, 0:3]
            points = np.matmul(poses[i], scan_hom.T).T[:, 0:3]
            
            # Load labels
            label = np.full((points.shape[0],), 0)  # Default to unlabeled
            label_file = seq_path + "labels/" + str(i).zfill(6)+'.label'
            if os.path.isfile(label_file):
                raw_labels = open_label(label_file)[valid_p]
                # Map to our reduced label set
                for orig_label, mapped_label in LABEL_MAPPING.items():
                    label[raw_labels == orig_label] = mapped_label
            
            # Combined validation mask
            valids = (
                np.all(points > lower_bound, axis=1) & 
                np.all(points < upper_bound, axis=1) & 
                (label > 0))  # Only keep labeled points
            
            all_points.append(points[valids])
            all_intensities.append(scan[valids, 3])
            all_labels.append(label[valids])
        
        if not all_points:
            return np.zeros((0, 3 + 1 + NUM_CLASSES + 3))  # xyz + intensity + one-hot + normals
        
        # Concatenate all collected data
        cloud = np.concatenate(all_points)
        intensities = np.concatenate(all_intensities)
        labels = np.concatenate(all_labels)
        
        # Return empty array if no valid points
        if cloud.shape[0] == 0:
            return np.zeros((0, 3 + 1 + NUM_CLASSES + 3))
        
        # Voxel downsampling with proper label handling
        voxel_size = self.config['grid']['voxel_size']
        voxel_coords = np.floor(cloud / voxel_size).astype(int)
        unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        # Initialize output arrays
        downsampled_points = np.zeros((len(unique_voxels), 3))
        downsampled_intensities = np.zeros(len(unique_voxels))
        downsampled_labels = np.zeros((len(unique_voxels), NUM_CLASSES))
        
        # Process each voxel
        for voxel_idx in range(len(unique_voxels)):
            mask = (inverse == voxel_idx)
            voxel_points = cloud[mask]
            voxel_intensities = intensities[mask]
            voxel_labels = labels[mask]
            
            # Average position and intensity
            downsampled_points[voxel_idx] = np.mean(voxel_points, axis=0)
            downsampled_intensities[voxel_idx] = np.mean(voxel_intensities)
            
            # Majority voting for one-hot labels
            if len(voxel_labels) > 0:
                label_counts = np.bincount(voxel_labels, minlength=NUM_CLASSES)
                majority_label = np.argmax(label_counts)
                downsampled_labels[voxel_idx, majority_label] = 1
        
        # Create Open3D point cloud for normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        
        # Skip normal estimation if too few points
        if len(downsampled_points) < 3:
            normals = np.zeros((len(downsampled_points), 3))
        else:
            try:
                # First estimate normals
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.config['grid']['normal_eigenvalue_radius'], 
                        max_nn=30
                    )
                )
                # Then orient them
                pcd.orient_normals_to_align_with_direction()
                normals = np.asarray(pcd.normals)
            except RuntimeError as e:
                print(f"Normal estimation failed: {e}")
                normals = np.zeros((len(downsampled_points), 3))
        
        # Combine all features
        sparse_points_features = np.hstack((
            downsampled_points,          # x, y, z (3)
            downsampled_intensities[:, np.newaxis],  # intensity (1)
            downsampled_labels,          # one-hot labels (NUM_CLASSES)
            normals                      # normals (3)
        ))
        
        print(f'#points {cloud.shape[0]} -> {downsampled_points.shape[0]}')
        return sparse_points_features

    def convert(self):
        time_very_start=time.time()
        folders=self.train_folders + self.valid_folders + self.test_folders
        for j, p in enumerate(folders):
            # print('pcu.path',p.split('/')[-2])
            out_dir=pcu.path(
                self.config['dataset']['data_folders']['grid_output'])+pcu.path(p.split('/')[-2])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            calibration=parse_calibration(p + "calib.txt")
            poses=parse_poses(p+"poses.txt", calibration)
            scan_files=[
                f for f in sorted(os.listdir(os.path.join(p, "velodyne")))
                if f.endswith(".bin")]
            key_poses_idx, key_poses, distance_matrix=getKeyPoses(
                poses, self.config['grid']['pose_distance'])
            print('kp shape', key_poses.shape)
            np.savetxt(out_dir+'key_poses.txt',
                       np.reshape(key_poses, (key_poses.shape[0], 16)))
            for i, idx in enumerate(key_poses_idx):
                bla=1
                time_start=time.time()
                sparse_points_features=self.sparsifieO3d(
                    poses, idx, p, distance_matrix).astype('float32')
                print('seq', j, 'from', len(folders),
                      'keypose', i, 'from', len(key_poses_idx))
                print('sparsifie time', time.time() - time_start)
                time_start=time.time()
                octree=octree_handler.Octree()
                points=sparse_points_features[:, :3]
                octree.setInput(points)
                eig_normals=octree.computeEigenvaluesNormal(
                    self.config['grid']['normal_eigenvalue_radius'])
                if sparse_points_features.shape[0] == 0:
                    print("Warning: empty point cloud passed to octree")
                    continue
                sparse_points_features=np.hstack(
                    (sparse_points_features, eig_normals))
                print('normal and eigenvalues estimation time',
                      time.time() - time_start)
                # print('sparse_points',sparse_points.shape)
                pcu.saveCloud2Binary(sparse_points_features, str(
                    i).zfill(6)+'.bin', out_dir)

                # time_start = time.time()
                # sparse_points = self.sparsifieVoxelGrid(poses, idx, p, distance_matrix)
                # sparse_points = self.sparsifieO3d(poses, idx, p, distance_matrix)
                # print('sparsifie2 time', time.time() - time_start)
                # print('sparse_points2',sparse_points.shape)
                # pcu.visPointCloud(sparse_points)
                # np.savetxt('testgrid.xyz',sparse_points)
                # return None
        print('convert time', time.time() - time_very_start)


if __name__ == "__main__":
    start_time=time.time()

    parser=argparse.ArgumentParser("./kitti2voxel.py")
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=False,
        default="/data",
        help='dataset folder containing all sequences in a folder called "sequences".',
    )
    parser.add_argument(
        '--arch_cfg', '-cfg',
        type=str,
        required=False,
        default='config/depoco_OHE.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )

    FLAGS, unparsed=parser.parse_known_args()
    yaml = YAML(typ='safe', pure=True)
    ARCH = yaml.load(open(FLAGS.arch_cfg, 'r'))
    #ARCH=yaml.safe_load(open(FLAGS.arch_cfg, 'r'))

    input_folder=FLAGS.dataset + '/sequences/00/'
    print('input_folder', input_folder)
    calib = input_folder + "calib.txt"
    calibration=parse_calibration(calib)
    poses=parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
    idx, keypose, d=getKeyPoses(poses, delta=ARCH["grid"]["pose_distance"])

    xy=np.asarray(poses)
    xy=xy[:, 0:2, -1]
    x=xy[:, 0]
    y=xy[:, 1]
    print(xy.shape)
    plt.figure
    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(xy[idx, 0], xy[idx, 1], 'xr')
    plt.axis('equal')

    plt.show()

    converter=Kitti2voxelConverter(ARCH)
    # converter.getMaxMinHeight() # -9 to 4
    converter.convert()




