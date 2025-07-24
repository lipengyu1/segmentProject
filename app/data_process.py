from flask import Blueprint, request, jsonify, session
import os
import logging
from utils.extract_frames import extract_frames

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

bp_data_process = Blueprint('data_process', __name__)

# 上传和处理文件夹配置
UPLOAD_FOLDER = 'datasets/'
bp_data_process.config = {'UPLOAD_FOLDER': UPLOAD_FOLDER}

@bp_data_process.route('/frameExtraction', methods=['POST'])
def frame_extraction():
    # 获取前台传入的帧数
    data = request.get_json()
    frame_count = data.get('frame')

    if not frame_count:
        logger.error("帧数为空")
        return jsonify({'error': '帧数不能为空'}), 400

    # 验证帧数格式（必须为正整数）
    try:
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise ValueError("帧数必须为正整数")
    except ValueError as e:
        logger.error(f"无效的帧数: {str(e)}")
        return jsonify({'error': f'无效的帧数: {str(e)}'}), 400

    # 获取项目名
    project_name = data.get('project_name')
    if not project_name:
        logger.error("项目名为空")
        return jsonify({'error': '项目名不能为空'}), 400

    # 验证项目名格式（仅允许字母、数字、下划线和连字符）
    project_name = ''.join(c for c in project_name if c.isalnum() or c in ['_', '-'])
    if not project_name:
        logger.error("项目名包含非法字符")
        return jsonify({'error': '项目名包含非法字符'}), 400

    # 检查项目文件夹是否存在
    project_path = os.path.join(bp_data_process.config['UPLOAD_FOLDER'], project_name)
    video_path = os.path.join(project_path, 'videos')
    if not os.path.exists(video_path):
        logger.error(f"项目视频文件夹未找到: {video_path}")
        return jsonify({'error': '项目视频文件夹不存在'}), 400

    # 创建存储帧的文件夹
    images_path = os.path.join(project_path, 'images')
    try:
        os.makedirs(images_path, exist_ok=True)
        logger.debug(f"确保图像存储文件夹存在: {images_path}")
    except Exception as e:
        logger.error(f"创建图像存储文件夹失败: {str(e)}")
        return jsonify({'error': f'创建图像存储文件夹失败: {str(e)}'}), 500

    # 调用抽帧函数
    try:
        successful_extractions, errors = extract_frames(video_path, images_path, frame_count)
    except Exception as e:
        logger.error(f"抽帧处理失败: {str(e)}")
        return jsonify({'error': f'抽帧处理失败: {str(e)}'}), 500

    if successful_extractions and not errors:
        logger.debug(f"成功抽取 {len(successful_extractions)} 个视频的帧")
        return jsonify({'message': f'成功从 {len(successful_extractions)} 个视频抽取帧', 'extractions': successful_extractions})
    elif successful_extractions:
        logger.debug(f"部分抽取成功，{len(successful_extractions)} 个视频，{len(errors)} 个错误")
        return jsonify({'message': '部分视频帧抽取成功', 'extractions': successful_extractions, 'errors': errors})
    else:
        logger.error(f"所有视频帧抽取失败: {errors}")
        return jsonify({'error': '所有视频帧抽取失败', 'errors': errors}), 400