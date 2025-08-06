import depoco.utils.point_cloud_utils as pcu
import argparse
from depoco.trainer import DepocoNetTrainer
import torch
from ruamel.yaml import YAML 
import numpy as np
import open3d as o3d
import os


label_color_map = {
    0: [255, 255, 255],
    -1: [255, 255, 0],          # road (yellow)
    1: [0, 255, 0],        #  sidewalk (green)
    2: [0, 0, 255],    #  parking(blue)
    3: [0, 255, 255],    #  building(sky blue)
    4: [255, 0, 255],     #  vehicles (pink)
    5: [0, 0, 0],          #  trunk(black)
    6: [255, 0, 0],        #  humans (red)
    7: [100, 0 ,100],      #  poles(purple)
    8: [0, 0, 100],        # outlier(dark blue)
    9: [100, 100, 100],    # terrain(grey) 
}

def add_color_to_points(points):
    colors = []
    #print(points[:,4])
    for point in points:
        if np.isnan(point[1]):
            label = -1  # or some fallback
        else:
            label = int(point[1])

        #label = int(point[4])  # label is at index 4
        # Get color from map, default to black if label not found
        bgr_color = label_color_map.get(label, [255, 0, 0])
        # Convert BGR to RGB
        rgb_color = [bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0]
        colors.append(rgb_color)
    return np.array(colors)

def add_color_to_points_decoder(points):
    colors = []
    #print(points[:,4])
    for point in points:
        if int(point[0]) == -1:
            label = -1  
        else:
            label = int(point[0])

        #label = int(point[4])  # label is at index 4
        # Get color from map, default to black if label not found
        bgr_color = label_color_map.get(label, [0, 0, 0])
        # Convert BGR to RGB
        rgb_color = [bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0]
        colors.append(rgb_color)
    return np.array(colors)


# Visualize the point cloud
def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x,y,z
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Set background to black for better contrast
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    # Run visualization
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./sample_net_trainer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='configitecture yaml cfg file. See /config/config for example',
    )
    parser.add_argument(
        '--number', '-n',
        type=int,
        default=5,
        help='Number of maps to visualize',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
 
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(open(FLAGS.config, 'r'))
    print('loaded yaml flags')
    trainer = DepocoNetTrainer(config)
    trainer.loadModel(best=True)
    print('initialized  trainer')
    
    for i, batch in enumerate(trainer.submaps.getOrderedTrainSet()):
        with torch.no_grad():
            print(batch)
            print("Before compression")
            points = batch['points'].detach().cpu().numpy()
            colors = add_color_to_points(batch['points_attributes'].detach().cpu().numpy())
            print(f'colors before: {colors}')
            visualize_point_cloud(points, colors)

            points_est, features, point_attribute, nr_emb_points = trainer.encodeDecode(batch)
            print("After compression")

            print(f'nr embedding points: {nr_emb_points}, points out: {points_est.shape[0]}')

            features = features * 150 #scaling
            colors_est = add_color_to_points_decoder(features.detach().cpu().numpy())
            print(f'Colors: {colors_est}')

            visualize_point_cloud(points_est.detach().cpu().numpy(), colors_est)

            
        if i+1 >= FLAGS.number:
            break
