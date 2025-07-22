from flask import Blueprint, request, jsonify, session, render_template
import os
import logging
import threading
import numpy as np

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from utils.reconstruct_images import reconstruct_images
from utils.reconstruct_images import cleanup_vggt
bp_reconstruct = Blueprint('reconstruct', __name__)
viser_data = {}
data_lock = threading.Lock()

@bp_reconstruct.route('/project', methods=['POST'])
def create_project():
    data = request.json
    project_name = data.get('project_name')
    if not project_name:
        logger.error("Project name is empty")
        return jsonify({'error': '项目名称不能为空'}), 400

    project_dir = os.path.join('datasets', project_name)
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'images'), exist_ok=True)
    session['project_name'] = project_name
    logger.debug(f"Project created: {project_name}")
    return jsonify({'message': f'项目 "{project_name}" 创建成功'})

@bp_reconstruct.route('/reconstruct', methods=['POST'])
def reconstruct():
    logger.debug("Reconstruct endpoint called")
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    image_folder = os.path.join('datasets', project_name, 'images')
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        logger.error(f"Image folder not found or empty: {image_folder}")
        return jsonify({'error': '图像文件夹为空或不存在'}), 400

    logger.debug(f"Processing images from: {image_folder}")
    result = reconstruct_images(image_folder)

    if isinstance(result, tuple) and "error" in result[0]:
        logger.error(f"Reconstruction error: {result[0]['error']}")
        return jsonify(result[0]), result[1]

    with data_lock:
        viser_data[project_name] = result

    logger.debug("Reconstruction successful")
    return jsonify({
        "message": "重建成功",
        "point_count": len(result.get("points_centered", []))
    })

@bp_reconstruct.route('/api/pointcloud', methods=['GET'])
def get_pointcloud():
    """获取点云数据"""
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    with data_lock:
        if project_name not in viser_data:
            logger.error("Point cloud data not generated")
            return jsonify({'error': '点云数据未生成'}), 404

        data = viser_data[project_name]
        points = np.array(data["points_centered"])
        colors = np.array(data["colors_flat"])
        conf = np.array(data["conf_flat"]) if "conf_flat" in data else np.ones(len(points))

        # 应用初始置信度过滤
        init_conf_threshold = 50.0
        threshold_val = np.percentile(conf, init_conf_threshold)
        conf_mask = (conf >= threshold_val) & (conf > 0.1)
        filtered_points = points[conf_mask]
        filtered_colors = colors[conf_mask]

        return jsonify({
            "points": filtered_points.tolist(),
            "colors": filtered_colors.tolist(),
            "cameras": data.get("cameras", []),
            "scene_center": data.get("scene_center", [0.0, 0.0, 0.0]),
            "point_count": len(filtered_points)
        })

@bp_reconstruct.route('/api/filter-points', methods=['POST'])
def filter_points():
    """动态过滤点云"""
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    data = request.json
    conf_percent = data.get('conf_percent', 50)
    frame_filter = data.get('frame', 'all')

    with data_lock:
        if project_name not in viser_data:
            return jsonify({'error': '点云数据未生成'}), 404

        full_data = viser_data[project_name]
        points = np.array(full_data["points_centered"])
        colors = np.array(full_data["colors_flat"])
        conf = np.array(full_data["conf_flat"])
        frame_indices = np.array(full_data["frame_indices"])

        threshold_val = np.percentile(conf, conf_percent)
        conf_mask = (conf >= threshold_val) & (conf > 0.1)

        if frame_filter != 'all':
            frame_idx = int(frame_filter)
            frame_mask = frame_indices == frame_idx
            combined_mask = conf_mask & frame_mask
        else:
            combined_mask = conf_mask

        filtered_points = points[combined_mask]
        filtered_colors = colors[combined_mask]

        return jsonify({
            "points": filtered_points.tolist(),
            "colors": filtered_colors.tolist(),
            "point_count": len(filtered_points)
        })

@bp_reconstruct.route('/api/frames', methods=['GET'])
def get_frames():
    """获取可用帧数"""
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    with data_lock:
        if project_name not in viser_data:
            logger.error("Point cloud data not generated")
            return jsonify({'error': '点云数据未生成'}), 404

        data = viser_data[project_name]
        frame_indices = np.array(data["frame_indices"])
        unique_frames = np.unique(frame_indices).tolist()

        return jsonify({
            "frames": unique_frames,
            "total_frames": len(unique_frames)
        })

@bp_reconstruct.route('/shutdown_vggt', methods=['POST'])
def shutdown_vggt():
    """关闭 VGGT 相关内容，仅保留 Flask"""
    global viser_data
    project_name = session.get('project_name')
    if project_name and project_name in viser_data:
        with data_lock:
            del viser_data[project_name]  # 清除项目数据
    cleanup_vggt()  # 清理 VGGT 资源
    logger.debug("VGGT shutdown completed, Flask remains active")
    return jsonify({"message": "VGGT 资源已关闭"})

@bp_reconstruct.route('/viewer')
def viewer():
    return render_template('viewer.html')