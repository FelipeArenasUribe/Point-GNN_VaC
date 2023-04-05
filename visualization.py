import open3d
import numpy as np

import time

from dataset.kitti_dataset import KittiDataset

from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn,\
    get_encoding_len

from models.graph_gen import get_graph_generate_fn

from collections import namedtuple

Points = namedtuple('Points', ['xyz', 'attr'])

dataset = KittiDataset(
    '/media/felipearur/ZackUP/dataset/kitti/image/training/image_2/',
    '/media/felipearur/ZackUP/dataset/kitti/velodyne/training/velodyne/',
    '/media/felipearur/ZackUP/dataset/kitti/calib/training/calib/',
    '',
    '/media/felipearur/ZackUP/dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

config = {
    "box_encoding_method": "classaware_all_class_box_encoding",
    "downsample_by_voxel_size": True,
    "eval_is_training": True,
    "graph_gen_kwargs": {
        "add_rnd3d": True,
        "base_voxel_size": 0.8,
        "downsample_method": "random",
        "level_configs": [
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 1.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 0,
                "graph_scale": 1
            },
            {
                "graph_gen_kwargs": {
                    "num_neighbors": 256,
                    "radius": 4.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 1,
                "graph_scale": 1
            }
        ]
    },
    "graph_gen_method": "multi_level_local_graph_v3",
    "input_features": "i",
    "label_method": "Car",
    "loss": {
        "cls_loss_type": "softmax",
        "cls_loss_weight": 0.1,
        "loc_loss_weight": 10.0
    },
    "model_kwargs": {
        "layer_configs": [
            {
                "graph_level": 0,
                "kwargs": {
                    "output_MLP_activation_type": "ReLU",
                    "output_MLP_depth_list": [
                        300,
                        300
                    ],
                    "output_MLP_normalization_type": "NONE",
                    "point_MLP_activation_type": "ReLU",
                    "point_MLP_depth_list": [
                        32,
                        64,
                        128,
                        300
                    ],
                    "point_MLP_normalization_type": "NONE"
                },
                "scope": "layer1",
                "type": "scatter_max_point_set_pooling"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer2",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer3",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "auto_offset": True,
                    "auto_offset_MLP_depth_list": [
                        64,
                        3
                    ],
                    "auto_offset_MLP_feature_activation_type": "ReLU",
                    "auto_offset_MLP_normalization_type": "NONE",
                    "edge_MLP_activation_type": "ReLU",
                    "edge_MLP_depth_list": [
                        300,
                        300
                    ],
                    "edge_MLP_normalization_type": "NONE",
                    "update_MLP_activation_type": "ReLU",
                    "update_MLP_depth_list": [
                        300,
                        300
                    ],
                    "update_MLP_normalization_type": "NONE"
                },
                "scope": "layer4",
                "type": "scatter_max_graph_auto_center_net"
            },
            {
                "graph_level": 1,
                "kwargs": {
                    "activation_type": "ReLU",
                    "normalization_type": "NONE"
                },
                "scope": "output",
                "type": "classaware_predictor"
            }
        ],
        "regularizer_kwargs": {
            "scale": 5e-07
        },
        "regularizer_type": "l1"
    },
    "model_name": "multi_layer_fast_local_graph_model_v2",
    "nms_overlapped_thres": 0.01,
    "num_classes": 4,
    "runtime_graph_gen_kwargs": {
        "add_rnd3d": False,
        "base_voxel_size": 0.8,
        "level_configs": [
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 1.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 0,
                "graph_scale": 0.5
            },
            {
                "graph_gen_kwargs": {
                    "num_neighbors": -1,
                    "radius": 4.0
                },
                "graph_gen_method": "disjointed_rnn_local_graph_v3",
                "graph_level": 1,
                "graph_scale": 0.5
            }
        ]
    }
}

BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

train_config = {
    "NUM_GPU": 2,
    "NUM_TEST_SAMPLE": -1,
    "batch_size": 4,
    "capacity": 1,
    "checkpoint_path": "model",
    "config_path": "config",
    "data_aug_configs": [
        {
            "method_kwargs": {
                "expend_factor": [
                    1.0,
                    1.0,
                    1.0
                ],
                "method_name": "normal",
                "yaw_std": 0.39269908169872414
            },
            "method_name": "random_rotation_all"
        },
        {
            "method_kwargs": {
                "flip_prob": 0.5
            },
            "method_name": "random_flip_all"
        },
        {
            "method_kwargs": {
                "appr_factor": 10,
                "expend_factor": [
                    1.1,
                    1.1,
                    1.1
                ],
                "max_overlap_num_allowed": 100,
                "max_overlap_rate": 0.01,
                "max_trails": 100,
                "method_name": "normal",
                "xyz_std": [
                    3,
                    0,
                    3
                ]
            },
            "method_name": "random_box_shift"
        }
    ],
    "decay_factor": 0.1,
    "decay_step": 400000,
    "gpu_memusage": -1,
    "initial_lr": 0.125,
    "load_dataset_every_N_time": 0,
    "load_dataset_to_mem": True,
    "max_epoch": 1718,
    "max_steps": 1400000,
    "num_load_dataset_workers": 16,
    "optimizer": "sgd",
    "optimizer_kwargs": {},
    "save_every_epoch": 20,
    "train_dataset": "train_car.txt",
    "train_dir": "./checkpoints/car_auto_T3_train",
    "unify_copies": True,
    "visualization": False
}

def Visualize_PC(nodes):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(nodes)
    pcd.paint_uniform_color([0.5, 1, 0.8])
    print(pcd)

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([pcd])

def Visualize_Graph(nodes, edges):
    colors = [[0.7, 0.8, 0.7] for i in range(len(edges))]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(nodes)
    line_set.lines = open3d.utility.Vector2iVector(edges)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(nodes)
    pcd.paint_uniform_color([0.5, 1, 0.8])
    print(pcd)

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        #opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([line_set, pcd])

def Visualize_VaC_Point_Clouds(Points):
    pcd = []
    #for i in range(0, len(Points)):
    #    pcd.append(open3d.PointCloud())
    #    pcd[i].points = open3d.Vector3dVector(Points[i])
    #    pcd[i].paint_uniform_color([rnd.random(), rnd.random(), rnd.random()])
    
    # Visualize original Point Cloud
    pcd.append(open3d.PointCloud())
    pcd[0].points = open3d.Vector3dVector(Points[0])
    pcd[0].paint_uniform_color([0, 0, 1])

    # Visualize downsampled Point Cloud
    pcd.append(open3d.PointCloud())
    pcd[1].points = open3d.Vector3dVector(Points[1])
    pcd[1].paint_uniform_color([1, 0, 0])

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.Visualizer()
        vis.create_window()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        #opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)
        print('Close graph to continue.')
        vis.run()  
        vis.destroy_window()

    custom_draw_geometry_load_option([*pcd])

def fetch_data(frame_idx):
    cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
        config['downsample_by_voxel_size'])
    cam_rgb_points_down = dataset.get_cam_points_in_image_with_rgb_downsampled(frame_idx,
        config['downsample_by_voxel_size'])
    
    
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cam_rgb_points_down.xyz, **config['graph_gen_kwargs'])
    '''
    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
            cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    '''
    input_v = cam_rgb_points.attr
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    '''
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list)
    '''
    return cam_rgb_points, cam_rgb_points_down, input_v, vertex_coord_list, keypoint_indices_list, edges_list

if __name__ == "__main__":
    for frame_idx in range(0, 10):
        #input_v, vertex_coord_list, keypoint_indices_list, edges_list = fetch_data(frame_idx)
        start = time.time()
        cam_rgb_points, cam_rgb_points_down, input_v, vertex_coord_list, keypoint_indices_list, edges_list = fetch_data(frame_idx)
        end = time.time()
        print('Pre-processing time: ',(end-start))

        print('Original Point cloud size: ', len(cam_rgb_points.xyz))
        print('Downsampled Point cloud size: ', len(cam_rgb_points_down.xyz))

        Points = [cam_rgb_points.xyz, cam_rgb_points_down.xyz]

        print('--------------------- Point Cloud Visualization ---------------------')
        Visualize_VaC_Point_Clouds(Points)

        nodes = vertex_coord_list[1]
        edges = edges_list[1]
        keypoint_indices = keypoint_indices_list[1]

        print('--------------------- Graph Visualization ---------------------')
        Visualize_Graph(nodes, edges)

        input('Press enter to continue...')