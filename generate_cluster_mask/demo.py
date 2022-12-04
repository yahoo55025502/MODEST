import scipy
import numpy as np
import hydra
import open3d as o3d
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from utils.pointcloud_utils import load_velo_scan, transform_points
from pre_compute_pp_score import get_relative_pose
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm.auto import tqdm

CURRENT_FRAME = 0
TRACKING_INFO = 1
NUMBER_STATUS = 2


class Cache:
    def __init__(self, args):
        self.idx = 0
        self.tracking_seq = 0
        self.pcd = o3d.geometry.PointCloud()
        self.track_list = pickle.load(open(args.track_path, "rb"))
        self.valid_idx = pickle.load(open(args.idx_info, "rb"))
        self.oxts_path = osp.join(args.data_root, "oxts")
        self.l2e_path = osp.join(args.data_root, "l2e")
        self.pp_score_path = args.pp_score_path
        self.data_root = args.data_root
        self.status = CURRENT_FRAME

    def get_l2e(self, name):
        return np.load(osp.join(self.l2e_path, f"{name:06d}.npy"))

    def get_pose(self, name):
        with open(osp.join(self.oxts_path, f"{name:06d}.txt"), "r") as f:
            info = np.array([float(x) for x in f.readline().split()])
            trans = np.eye(4)
            trans[:3, 3] = info[:3]
            trans[:3, :3] = R.from_euler('xyz', info[3:]).as_matrix()
            pose = trans.astype(np.float32)
        return pose


def update_cache(vis):
    global data_cache
    if data_cache.status == CURRENT_FRAME:
        name = list(data_cache.valid_idx.keys())[data_cache.idx]
        seq_id, frame = data_cache.valid_idx[name][:2]
        print(
            f"Currently rendering seq/frame: {seq_id}/{frame}")
        data_cache.pcd.points = o3d.utility.Vector3dVector(load_velo_scan(
            osp.join(data_cache.data_root, "velodyne", f"{name:06d}.bin"))[:, :3])
        # data_cache.pcd.paint_uniform_color(np.array([0., 0., 0.]))
        pp_score = np.load(
            osp.join(data_cache.pp_score_path, f"{name:06d}.npy"))
        cmap = plt.get_cmap('jet')
        colors = cmap(pp_score)[:, :3]
        # pp_score = np.array([pp_score.T, pp_score.T, pp_score.T]).T
        # N = pp_score.shape[0]
        # red = np.tile(np.array([1., 0., 0.]), (N, 1))
        # blue = np.tile(np.array([0., 0., 1.]), (N, 1))
        # colors = blue + (red-blue)*pp_score
        data_cache.pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(data_cache.pcd)
    if data_cache.status == TRACKING_INFO:
        name = list(data_cache.valid_idx.keys())[data_cache.idx]
        # Collecting tracking PCDs
        num_seq = len(data_cache.valid_idx[name][2])
        print(
            f"Collecting Tracking information of seq/total_seq {data_cache.tracking_seq}/{num_seq-1}...")
        seq_id, indices = data_cache.valid_idx[name][2][data_cache.tracking_seq]
        fixed_l2e = data_cache.get_l2e(name)
        fixed_pose = data_cache.get_pose(name)
        # fixed_l2e = data_cache.get_l2e(
        #     data_cache.track_list[seq_id][indices[0]])
        # fixed_pose = data_cache.get_pose(
        #     data_cache.track_list[seq_id][indices[0]])
        combined_ptcs = []
        for frame in tqdm(indices):
            other_name = data_cache.track_list[seq_id][frame]
            _ptc = load_velo_scan(
                osp.join(
                    data_cache.data_root,
                    "velodyne",
                    f"{other_name:06d}.bin"))[:, :3]
            _relative_pose = get_relative_pose(
                fixed_l2e=fixed_l2e, fixed_ego=fixed_pose,
                # query_l2e=l2es[seq_id][frame], query_ego=poses[seq_id][frame],
                query_l2e=data_cache.get_l2e(other_name), query_ego=data_cache.get_pose(other_name),
                KITTI2NU=Quaternion(axis=(0, 0, 1), angle=np.pi).transformation_matrix)
            _ptc = transform_points(_ptc, _relative_pose)
            combined_ptcs.append(_ptc)
        combined_ptcs = np.concatenate(combined_ptcs).astype(np.float32)
        data_cache.pcd.points = o3d.utility.Vector3dVector(combined_ptcs)
        data_cache.pcd.paint_uniform_color(np.array([0., 0., 0.]))
        vis.update_geometry(data_cache.pcd)


def draw_scenes(args):
    def update_to_next_frame(vis):
        if data_cache.status == CURRENT_FRAME:
            data_cache.idx += 1
        if data_cache.status == TRACKING_INFO:
            name = list(data_cache.valid_idx.keys())[data_cache.idx]
            num_seq = len(data_cache.valid_idx[name][2])
            data_cache.tracking_seq = (data_cache.tracking_seq+1) % num_seq
        update_cache(vis)
        return False

    def switch_status(vis):
        print(
            f"Switching visualized contents from {data_cache.status} to {(data_cache.status + 1) % NUMBER_STATUS}")
        data_cache.status = (data_cache.status + 1) % NUMBER_STATUS
        if data_cache.status == TRACKING_INFO:
            data_cache.tracking_seq = 0
        update_cache(vis)
        return False

    # Setup visualization variables
    global data_cache
    data_cache = Cache(args)

    data_cache.pcd.points = o3d.utility.Vector3dVector(load_velo_scan(
        osp.join(
            args.data_root,
            "velodyne",
            f"{next(iter(data_cache.valid_idx.keys())):06d}.bin"))[:, :3])
    data_cache.pcd.paint_uniform_color(np.array([0., 0., 0.]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(data_cache.pcd)
    vis.get_view_control().set_zoom(0.08)
    vis.get_view_control().set_lookat(np.array([0., 0., 0.]))
    vis.get_render_option().point_size = 1.0
    vis.register_key_callback(ord("N"), update_to_next_frame)
    vis.register_key_callback(ord("S"), switch_status)
    vis.run()
    vis.destroy_window()


# def draw_scenes(args):
#     track_list = pickle.load(open(args.track_path, "rb"))
#     valid_idx = pickle.load(open(args.idx_info, "rb"))
#     oxts_path = osp.join(args.data_root, "oxts")
#     l2e_path = osp.join(args.data_root, "l2e")
#     examined_train_idx = list(valid_idx.keys())[args.examine_idx]
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(load_velo_scan(
#         osp.join(
#             args.data_root,
#             "velodyne",
#             f"{track_list[valid_idx[examined_train_idx][0]][valid_idx[examined_train_idx][1]]:06d}.bin"))[:, :3])
#     pcd.paint_uniform_color(np.array([0., 0., 0.]))
#     vis.add_geometry(pcd)
#     vis.get_view_control().set_zoom(0.08)
#     vis.get_view_control().set_lookat(np.array([0., 0., 0.]))
#     vis.get_render_option().point_size = 1.0

#     for train_idx in valid_idx.keys():
#         pcd.points = o3d.utility.Vector3dVector(load_velo_scan(
#             osp.join(
#                 args.data_root,
#                 "velodyne",
#                 f"{train_idx:06d}.bin"))[:, :3])
#         pcd.paint_uniform_color(np.array([0., 0., 0.]))
#         vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()

#     vis.destroy_window()


@hydra.main(version_base=None, config_path="configs/", config_name="demo_config.yaml")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    draw_scenes(args)


if __name__ == "__main__":
    main()
