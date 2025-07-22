import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import logging
import trimesh
import pycolmap
import gc

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 全局变量跟踪 VGGT 资源，与 reconstruct_images.py 保持一致
vggt_model = None
vggt_device = None
vggt_resources = {}  # 存储加载的函数和资源

def load_vggt_model():
    """加载 VGGT 模型"""
    global vggt_model, vggt_device, vggt_resources
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
        from vggt.dependency.track_predict import predict_tracks
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

        vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {vggt_device}")
        vggt_model = VGGT()
        vggt_model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
        vggt_model.eval()
        vggt_model = vggt_model.to(vggt_device)

        vggt_resources = {
            "model": vggt_model,
            "load_and_preprocess_images_square": load_and_preprocess_images_square,
            "pose_encoding_to_extri_intri": pose_encoding_to_extri_intri,
            "unproject_depth_map_to_point_map": unproject_depth_map_to_point_map,
            "create_pixel_coordinate_grid": create_pixel_coordinate_grid,
            "randomly_limit_trues": randomly_limit_trues,
            "predict_tracks": predict_tracks,
            "batch_np_matrix_to_pycolmap": batch_np_matrix_to_pycolmap,
            "batch_np_matrix_to_pycolmap_wo_track": batch_np_matrix_to_pycolmap_wo_track
        }
        return True
    except Exception as e:
        logger.error(f"Failed to load VGGT model: {str(e)}")
        return False

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def cleanup_vggt():
    """清理 VGGT 相关资源"""
    global vggt_model, vggt_device, vggt_resources
    try:
        if vggt_model is not None:
            if vggt_device.startswith("cuda"):
                vggt_model.cpu()  # 移动到 CPU 再清理
                torch.cuda.empty_cache()  # 释放 GPU 显存
                gc.collect()  # 强制垃圾回收
            del vggt_model  # 删除引用
            vggt_model = None
        vggt_resources.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 再次清理残留
        gc.collect()
        logger.debug("VGGT resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to cleanup VGGT resources: {str(e)}")

def export_to_colmap(image_folder, output_dir):
    global vggt_model, vggt_device, vggt_resources
    # 默认参数，参考 demo_colmap.py
    seed = 42
    use_ba = False
    max_reproj_error = 8.0
    shared_camera = False
    camera_type = "PINHOLE"
    vis_thresh = 0.2
    query_frame_num = 8
    max_query_pts = 4096
    fine_tracking = True
    conf_thres_value = 5.0
    img_load_resolution = 1024
    vggt_fixed_resolution = 518

    # 设置种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}, dtype: {dtype}")

    # 加载模型
    if vggt_model is None:
        if not load_vggt_model():
            return {'error': 'Failed to load VGGT model'}

    model = vggt_resources["model"]
    load_and_preprocess_images_square = vggt_resources["load_and_preprocess_images_square"]
    pose_encoding_to_extri_intri = vggt_resources["pose_encoding_to_extri_intri"]
    unproject_depth_map_to_point_map = vggt_resources["unproject_depth_map_to_point_map"]
    create_pixel_coordinate_grid = vggt_resources["create_pixel_coordinate_grid"]
    randomly_limit_trues = vggt_resources["randomly_limit_trues"]
    predict_tracks = vggt_resources["predict_tracks"]
    batch_np_matrix_to_pycolmap = vggt_resources["batch_np_matrix_to_pycolmap"]
    batch_np_matrix_to_pycolmap_wo_track = vggt_resources["batch_np_matrix_to_pycolmap_wo_track"]

    # 获取图像路径
    image_path_list = glob.glob(os.path.join(image_folder, "*"))
    if len(image_path_list) == 0:
        cleanup_vggt()  # 清理资源
        return {'error': f"No images found in {image_folder}"}
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # 加载并预处理图像
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    logger.debug(f"Loaded {len(images)} images from {image_folder}")

    # 运行 VGGT
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=fine_tracking,
            )
            torch.cuda.empty_cache()

        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            cleanup_vggt()  # 清理资源
            return {'error': "No reconstruction can be built with BA"}

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = 5.0
        max_points_for_colmap = 100000
        shared_camera = False
        camera_type = "PINHOLE"

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        logger.debug("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    # 重命名和调整相机参数
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # 保存重建结果
    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write(output_dir)
    logger.debug(f"Saving reconstruction to {output_dir}")

    # 保存点云
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(output_dir, "points.ply"))
    logger.debug(f"Saved point cloud to {os.path.join(output_dir, 'points.ply')}")

    # 清理资源，模仿 reconstruct_images.py
    del images, original_coords, extrinsic, intrinsic, depth_map, depth_conf, points_3d, points_rgb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    cleanup_vggt()  # 显式调用清理函数

    return {'message': '导出完成', 'output_path': output_dir}

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp
            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction