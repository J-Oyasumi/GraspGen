# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import json
import os

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a single object point cloud after GraspGen inference"
    )
    parser.add_argument(
        "--ply_file",
        type=str,
        default="",
        help="Path to a single PLY point cloud file for inference (optional)",
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="/code/realrobot_pc/final/",
        help="Directory containing JSON files with point cloud data (used when --ply_file is not set)",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="",
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "-o","--output_dir",
        type=str,
        default="",
        help="If set, save grasp poses as .npy to this directory (centered and original frame)",
    )

    return parser.parse_args()


def load_ply_point_cloud(ply_path):
    """Load point cloud from PLY file. Returns pc (Nx3), pc_color (Nx3) in 0-255."""
    geom = trimesh.load(ply_path)
    # 支持 trimesh.PointCloud 或 trimesh.Trimesh（用顶点当点云）
    if isinstance(geom, trimesh.PointCloud):
        pc = np.asarray(geom.vertices, dtype=np.float64)
        if hasattr(geom, "colors") and geom.colors is not None:
            colors = np.asarray(geom.colors)
            if colors.shape[1] == 4:
                colors = colors[:, :3]
            pc_color = np.clip(colors.astype(np.float64), 0, 255).astype(np.uint8)
        else:
            pc_color = np.full((len(pc), 3), 128, dtype=np.uint8)
        return pc, pc_color
    elif isinstance(geom, trimesh.Trimesh):
        pc = np.asarray(geom.vertices, dtype=np.float64)
        if hasattr(geom.visual, "vertex_colors") and geom.visual.vertex_colors is not None:
            colors = np.asarray(geom.visual.vertex_colors)
            if colors.shape[1] == 4:
                colors = colors[:, :3]
            pc_color = np.clip(colors.astype(np.float64), 0, 255).astype(np.uint8)
        else:
            pc_color = np.full((len(pc), 3), 128, dtype=np.uint8)
        return pc, pc_color
    elif isinstance(geom, trimesh.Scene):
        geoms = list(geom.geometry.values())
        if not geoms:
            raise ValueError(f"PLY scene has no geometry: {ply_path}")
        return load_ply_point_cloud_from_geom(geoms[0])
    else:
        return load_ply_point_cloud_from_geom(geom)


def load_ply_point_cloud_from_geom(geom):
    """从单个几何体得到点云与颜色（供 Scene 递归用）。"""
    if isinstance(geom, trimesh.PointCloud):
        pc = np.asarray(geom.vertices, dtype=np.float64)
        if hasattr(geom, "colors") and geom.colors is not None:
            c = np.asarray(geom.colors)[:, :3]
            pc_color = np.clip(c.astype(np.float64), 0, 255).astype(np.uint8)
        else:
            pc_color = np.full((len(pc), 3), 128, dtype=np.uint8)
        return pc, pc_color
    if isinstance(geom, trimesh.Trimesh):
        pc = np.asarray(geom.vertices, dtype=np.float64)
        if hasattr(geom.visual, "vertex_colors") and geom.visual.vertex_colors is not None:
            c = np.asarray(geom.visual.vertex_colors)[:, :3]
            pc_color = np.clip(c.astype(np.float64), 0, 255).astype(np.uint8)
        else:
            pc_color = np.full((len(pc), 3), 128, dtype=np.uint8)
        return pc, pc_color
    raise ValueError(f"Unsupported geometry type: {type(geom)}")


def process_point_cloud(pc, grasps, grasp_conf):
    """Process point cloud and grasps by centering them."""
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}")

    # Ensure grasps have correct homogeneous coordinate
    grasps[:, 3, 3] = 1

    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    return pc_centered, grasps_centered, scores


def run_inference_on_point_cloud(pc_centered, pc_color, grasp_sampler, gripper_name, args, vis):
    """Run outlier removal, inference, and visualization. Returns (grasps_centered, grasp_conf) or (None, None) if no grasps."""
    visualize_pointcloud(vis, "pc", pc_centered, pc_color, size=0.0025)

    pc_filtered, pc_removed = point_cloud_outlier_removal(
        torch.from_numpy(pc_centered)
    )
    pc_filtered = pc_filtered.numpy()
    pc_removed = pc_removed.numpy()
    visualize_pointcloud(vis, "pc_removed", pc_removed, [255, 0, 0], size=0.003)

    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps_inferred) > 0:
        grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
        grasps_inferred = grasps_inferred.cpu().numpy()
        grasps_inferred[:, 3, 3] = 1
        scores_inferred = get_color_from_score(
            grasp_conf_inferred, use_255_scale=True
        )
        print(
            f"Inferred {len(grasps_inferred)} grasps, with scores ranging from {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
        )
        for j, grasp in enumerate(grasps_inferred):
            visualize_grasp(
                vis,
                f"grasps_objectpc_filtered/{j:03d}/grasp",
                grasp,
                color=scores_inferred[j],
                gripper_name=gripper_name,
                linewidth=0.6,
            )
        return grasps_inferred, grasp_conf_inferred
    else:
        print("No grasps found from inference! Skipping to next object...")
        return None, None


def save_grasps_npy(output_dir, name_prefix, grasps_centered, grasp_conf, T_centered_to_original):
    """Save grasp poses in centered and original frame as .npy, plus confidence scores."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(grasps_centered)} grasp poses to {output_dir}")
    print(f"Grasp poses shape: {grasps_centered.shape}")
    print(f"Grasp conf shape: {grasp_conf.shape}")
    np.save(os.path.join(output_dir, "grasps_centered.npy"), grasps_centered)
    np.save(os.path.join(output_dir, "grasp_scores.npy"), grasp_conf)
    grasps_original = np.array(
        [T_centered_to_original @ g for g in grasps_centered.tolist()],
        dtype=np.float64,
    )
    grasps_original[:, 3, 3] = 1
    np.save(os.path.join(output_dir, "grasps_original.npy"), grasps_original)

    if grasp_conf is not None and len(grasp_conf) > 0:
        best_idx = int(np.argmax(grasp_conf))
        best_grasp_centered = grasps_centered[best_idx]
        best_grasp_original = grasps_original[best_idx]
        np.save(os.path.join(output_dir, "best_grasp_centered.npy"), best_grasp_centered)
        np.save(os.path.join(output_dir, "best_grasp_original.npy"), best_grasp_original)


if __name__ == "__main__":
    args = parse_args()

    if args.gripper_config == "":
        raise ValueError("Gripper config is required")

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    if not args.ply_file and not os.path.isdir(args.sample_data_dir):
        raise ValueError(
            "Either --ply_file must be set to a PLY file, or --sample_data_dir must be an existing directory."
        )

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    vis = create_visualizer()
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    if args.ply_file:
        # 单 PLY 文件推理
        if not os.path.isfile(args.ply_file):
            raise ValueError(f"PLY file not found: {args.ply_file}")
        print(f"Processing PLY: {args.ply_file}")
        vis.delete()
        pc, pc_color = load_ply_point_cloud(args.ply_file)
        T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
        pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
        grasps_centered, grasp_conf = run_inference_on_point_cloud(
            pc_centered, pc_color, grasp_sampler, gripper_name, args, vis
        )
        if args.output_dir and grasps_centered is not None:
            T_centered_to_original = tra.inverse_matrix(T_subtract_pc_mean)
            save_grasps_npy(
                args.output_dir, "",  # name_prefix 不再使用
                grasps_centered, grasp_conf, T_centered_to_original,
            )
            print(f"Saved grasps to {args.output_dir}")
        input("Press Enter to exit...")
    else:
        # 原逻辑：目录下多个 JSON
        json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))
        for json_file in json_files:
            print(f"Processing {json_file}")
            vis.delete()

            data = json.load(open(json_file, "rb"))
            pc = np.array(data["pc"])
            pc_color = np.array(data["pc_color"])
            grasps = np.array(data["grasp_poses"])
            grasp_conf = np.array(data["grasp_conf"])

            pc_centered, grasps_centered, scores = process_point_cloud(
                pc, grasps, grasp_conf
            )
            T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
            grasps_inferred, grasp_conf_inferred = run_inference_on_point_cloud(
                pc_centered, pc_color, grasp_sampler, gripper_name, args, vis
            )
            if args.output_dir and grasps_inferred is not None:
                T_centered_to_original = tra.inverse_matrix(T_subtract_pc_mean)
                save_grasps_npy(
                    args.output_dir, "",
                    grasps_inferred, grasp_conf_inferred, T_centered_to_original,
                )
                print(f"Saved grasps to {args.output_dir}")
            input("Press Enter to continue to next object...")
