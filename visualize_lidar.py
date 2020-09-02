import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import open3d as o3d

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

FILENAME = "/media/john/storage/waymo/training_0000/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

pcd = o3d.geometry.PointCloud()

visualizer = o3d.visualization.Visualizer()
visualizer.create_window("visualizer", 1920, 1080, 0, 0)
visualizer.add_geometry(pcd)

view = visualizer.get_view_control()
camera_parameters = o3d.io.read_pinhole_camera_parameters("view_point.json")
visualizer.reset_view_point(True)
view.convert_from_pinhole_camera_parameters(camera_parameters)

render_options = visualizer.get_render_option()
render_options.background_color = [0, 0, 0]
render_options.point_size = 0.5 * render_options.point_size 

#this is for working around a bug in open3D with setting the view port
view_set = False

frame_index = 0
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    points_all = np.concatenate(points, axis=0)
  
    pcd.points = o3d.utility.Vector3dVector(points_all)

    visualizer.update_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()
    
    visualizer.capture_screen_image("lidar_pngs/{:03d}.png".format(frame_index))

    frame_index += 1

    #hack to fix open3d view point
    if not view_set:
      visualizer.reset_view_point(True)
      view.convert_from_pinhole_camera_parameters(camera_parameters)
      view_set = True
      

visualizer.destroy_window()
