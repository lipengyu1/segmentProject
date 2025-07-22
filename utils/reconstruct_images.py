import os
import glob
import time
import threading
import numpy as np
import torch
import gc
from tqdm.auto import tqdm
import requests
from flask import redirect, jsonify
import logging
import cv2
import base64


from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
import viser.transforms as viser_tf

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    import onnxruntime
except ImportError:
    logger.warning("onnxruntime not found. Sky segmentation may not work.")

# 全局变量跟踪 VGGT 资源
vggt_model = None
vggt_device = None
vggt_resources = {}  # 存储加载的函数和资源


def load_vggt_model():
    """加载 VGGT 模型"""
    global vggt_model, vggt_device, vggt_resources
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        import viser
        import viser.transforms as viser_tf
        from models.vggt.visual_util import segment_sky, download_file_from_url

        vggt_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {vggt_device}")
        vggt_model = VGGT()
        vggt_model.load_state_dict(
            torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
        vggt_model.eval()
        vggt_model = vggt_model.to(vggt_device)

        vggt_resources = {
            "model": vggt_model,
            "load_and_preprocess_images": load_and_preprocess_images,
            "unproject_depth_map_to_point_map": unproject_depth_map_to_point_map,
            "pose_encoding_to_extri_intri": pose_encoding_to_extri_intri,
            "viser": viser,
            "viser_tf": viser_tf,
            "segment_sky": segment_sky,
            "download_file_from_url": download_file_from_url
        }
        return (vggt_model, vggt_device, load_and_preprocess_images, unproject_depth_map_to_point_map,
                pose_encoding_to_extri_intri, viser, viser_tf, segment_sky, download_file_from_url)
    except Exception as e:
        logger.error(f"Failed to load VGGT model: {str(e)}")
        raise


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


def apply_sky_segmentation(conf: np.ndarray, image_folder: str, segment_sky, download_file_from_url) -> np.ndarray:
    """应用天空分割到置信度分数，参考 demo_viser.py 的实现"""
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skyseg_path = os.path.join(project_root, "skyseg.onnx")

    # 检查 skyseg.onnx 是否存在
    if not os.path.exists(skyseg_path):
        logger.error(
            "skyseg.onnx not found. Please manually download it from https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx and place it in the project directory.")
        raise RuntimeError("skyseg.onnx not found")

    # 验证文件有效性
    if os.path.exists(skyseg_path) == 0:
        logger.error("skyseg.onnx file is empty. Please ensure a valid file is provided.")
        raise RuntimeError("skyseg.onnx file is empty")

    try:
        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        logger.debug("Successfully loaded skyseg.onnx into InferenceSession")
    except Exception as e:
        logger.error(f"Failed to load skyseg.onnx into InferenceSession: {str(e)}")
        raise RuntimeError(f"Failed to load skyseg.onnx: {str(e)}")

    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    logger.debug("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S], desc="Processing sky masks")):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            logger.debug(f"Loaded cached sky mask: {mask_filepath}")
        else:
            try:
                sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
                logger.debug(f"Generated sky mask for {image_name}")
            except Exception as e:
                logger.error(f"Failed to generate sky mask for {image_name}: {str(e)}")
                sky_mask = None

        if sky_mask is not None:
            # 调整掩码大小以匹配 H×W
            if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                sky_mask = cv2.resize(sky_mask, (W, H))
            sky_mask_list.append(sky_mask)
        else:
            logger.warning(f"Skipping invalid sky mask for {image_name}")
            sky_mask_list.append(np.zeros((H, W), dtype=np.uint8))  # 使用零掩码作为回退

    if not sky_mask_list:
        logger.error("No valid sky masks generated")
        raise RuntimeError("Failed to generate any sky masks")

    # 转换为形状为 S×H×W 的 numpy 数组
    sky_mask_array = np.array(sky_mask_list)
    # 将天空掩码应用于置信度分数
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    logger.debug("Sky segmentation applied successfully")
    return conf


def prepare_viser_data(
        pred_dict: dict,
        init_conf_threshold: float = 25.0,
        use_point_map: bool = False,
        mask_sky: bool = True,
        image_folder: str = None,
        segment_sky=None,
        download_file_from_url=None
) -> dict:
    """准备点云数据供前端使用"""
    # 解包预测字典
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # 如果不使用预计算点云，则从深度图计算世界点
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # 如果启用天空分割
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder, segment_sky, download_file_from_url)

    # 将图像从 (S, 3, H, W) 转换为 (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # 现在为 (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # 展平数据
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)  # 确保范围 0-255
    # logger.debug(f"Colors flat shape: {colors_flat.shape}, sample: {colors_flat[:10]}")  # 调试日志
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # 形状通常为 (S, 4, 4)
    cam_to_world = cam_to_world_mat[:, :3, :]  # 仅存储 (3,4) 部分

    # 计算场景中心并重新居中
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # 存储帧索引以便按帧过滤
    frame_indices = np.repeat(np.arange(S), H * W)

    # 准备相机数据
    cameras = []
    for i in range(S):
        position = cam_to_world[i, :, -1].tolist()
        rotation = viser_tf.SO3.from_matrix(cam_to_world[i, :, :3]).wxyz.tolist()

        # 转换图像为base64
        img = (images[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        cameras.append({
            "position": position,
            "rotation": rotation,
            "image": img_base64,
            "fov": 2 * np.arctan2(H / 2, intrinsics_cam[i, 0, 0]),  # 计算FOV
            "aspect": W / H
        })

    # 计算初始置信度阈值
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)

    return {
        "points": points_centered[init_conf_mask].tolist(),
        "colors": colors_flat[init_conf_mask].tolist(),
        "conf_flat": conf_flat.tolist(),
        "points_centered": points_centered.tolist(),  # 所有点（用于动态过滤）
        "colors_flat": colors_flat.tolist(),  # 所有颜色
        "cameras": cameras,
        "scene_center": scene_center.tolist(),
        "frame_indices": frame_indices.tolist()
    }


def reconstruct_images(image_folder):
    """处理图像并生成重建数据，返回点云数据，参考 demo_viser.py 的实现"""
    logger.debug(f"Processing images from folder: {image_folder}")
    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    if not image_names:
        logger.error("No images found in the folder")
        return {"error": "No images found in the folder"}, 400

    try:
        # 加载模型和工具函数
        if vggt_model is None:
            load_vggt_model()
        model, device, load_and_preprocess_images, unproject_depth_map_to_point_map, pose_encoding_to_extri_intri, viser, viser_tf, segment_sky, download_file_from_url = (
            vggt_resources["model"], vggt_device, vggt_resources["load_and_preprocess_images"],
            vggt_resources["unproject_depth_map_to_point_map"], vggt_resources["pose_encoding_to_extri_intri"],
            vggt_resources["viser"], vggt_resources["viser_tf"], vggt_resources["segment_sky"],
            vggt_resources["download_file_from_url"]
        )
        logger.debug(f"Loading images from {image_folder}...")
        logger.debug(f"Found {len(image_names)} images")

        images = load_and_preprocess_images(image_names).to(device)
        logger.debug(f"Preprocessed images shape: {images.shape}")

        logger.debug("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        logger.debug("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        logger.debug("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                np_array = predictions[key].cpu().numpy()
                # 参考 demo_viser.py，移除 batch 维度
                if np_array.ndim > 1 and np_array.shape[0] == 1:
                    np_array = np_array.squeeze(0)
                predictions[key] = np_array

        logger.debug(f"extrinsic shape: {predictions['extrinsic'].shape}")
        logger.debug(f"intrinsic shape: {predictions['intrinsic'].shape}")

        # 处理 depth_map，保持 (S, H, W, 1) 形状
        depth_map = predictions["depth"]
        logger.debug(f"Original depth_map shape: {depth_map.shape}")

        # 参考 demo_viser.py，确保 depth_map 为 (S, H, W, 1)
        if depth_map.ndim == 5 and depth_map.shape[0] == 1:  # (1, S, H, W, 1)
            depth_map = depth_map.squeeze(0)  # 变为 (S, H, W, 1)
        elif depth_map.ndim == 3:  # (S, H, W)
            depth_map = depth_map[..., np.newaxis]  # 变为 (S, H, W, 1)
        elif depth_map.ndim == 4 and depth_map.shape[-1] != 1:  # (S, 1, H, W)
            logger.error(f"Unsupported depth_map shape: {depth_map.shape}")
            return {"error": f"Unsupported depth_map shape: {depth_map.shape}"}, 400
        elif depth_map.ndim != 4:  # 其他意外形状
            logger.error(f"Unsupported depth_map shape: {depth_map.shape}")
            return {"error": f"Unsupported depth_map shape: {depth_map.shape}"}, 400

        logger.debug(f"Processed depth_map shape: {depth_map.shape}")

        # 确保视图数匹配
        num_views = depth_map.shape[0]

        # 修复 extrinsic 和 intrinsic 的维度问题
        if predictions['extrinsic'].shape[0] != num_views:
            if predictions['extrinsic'].shape[0] == 1:
                logger.warning(f"Extrinsic has only 1 view, replicating for {num_views} views")
                predictions['extrinsic'] = np.repeat(predictions['extrinsic'], num_views, axis=0)
            else:
                logger.error(
                    f"Mismatch in number of views: depth_map has {num_views}, but extrinsic has {predictions['extrinsic'].shape[0]}")
                return {
                    "error": f"Mismatch in number of views: depth_map has {num_views}, but extrinsic has {predictions['extrinsic'].shape[0]}"}, 400

        if predictions['intrinsic'].shape[0] != num_views:
            if predictions['intrinsic'].shape[0] == 1:
                logger.warning(f"Intrinsic has only 1 view, replicating for {num_views} views")
                predictions['intrinsic'] = np.repeat(predictions['intrinsic'], num_views, axis=0)
            else:
                logger.error(
                    f"Mismatch in number of views: depth_map has {num_views}, but intrinsic has {predictions['intrinsic'].shape[0]}")
                return {
                    "error": f"Mismatch in number of views: depth_map has {num_views}, but intrinsic has {predictions['intrinsic'].shape[0]}"}, 400

        # 计算世界点，参考 demo_viser.py 直接传递 depth_map
        logger.debug("Computing world points...")
        try:
            world_points = unproject_depth_map_to_point_map(
                depth_map,
                predictions['extrinsic'],
                predictions['intrinsic']
            )
            logger.debug(f"World points shape: {world_points.shape}")
        except Exception as e:
            logger.error(f"Error computing world points: {str(e)}")
            return {"error": f"Failed to generate world points: {str(e)}"}, 400

        # 更新预测字典
        predictions["world_points"] = world_points
        predictions["world_points_conf"] = predictions["depth_conf"]
        predictions["depth"] = depth_map

        logger.debug("Preparing viser data...")
        viser_data = prepare_viser_data(
            predictions,
            init_conf_threshold=25.0,
            use_point_map=False,
            mask_sky=True,
            image_folder=image_folder,
            segment_sky=segment_sky,
            download_file_from_url=download_file_from_url
        )

        # 清理推理过程中的张量
        # del images, predictions
        # if vggt_device.startswith("cuda"):
        #     torch.cuda.empty_cache()
        # gc.collect()
        # 清理所有中间张量
        del images, predictions, depth_map, world_points
        if vggt_device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        logger.debug("Reconstruction data prepared successfully")
        return viser_data

    except Exception as e:
        logger.error(f"Reconstruction failed: {str(e)}", exc_info=True)
        return {"error": f"Reconstruction failed: {str(e)}"}, 500