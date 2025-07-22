from flask import Blueprint, request, jsonify, session
import os
import logging
import threading

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from utils.export_colmap import export_to_colmap

bp_export = Blueprint('export', __name__)
viser_data = {}
data_lock = threading.Lock()

@bp_export.route('/export', methods=['GET'])
def export():
    logger.debug("Export endpoint called")
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    image_folder = os.path.join('datasets', project_name, 'images')
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        logger.error(f"Image folder not found or empty: {image_folder}")
        return jsonify({'error': '图像文件夹为空或不存在'}), 400

    output_dir = os.path.join('datasets', project_name, 'sparse')
    try:
        result = export_to_colmap(image_folder, output_dir)
        if 'error' in result:
            logger.error(f"Export error: {result['error']}")
            return jsonify(result), 500
        logger.debug(f"Export successful to {output_dir}")
        return jsonify({'message': f'导出成功至 {output_dir}', 'output_path': output_dir})
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'导出失败: {str(e)}'}), 500
    finally:
        # 确保在任何情况下都清理资源，模仿 reconstruct.py 的 shutdown_vggt 逻辑
        from utils.reconstruct_images import cleanup_vggt  # 动态导入以避免循环引用
        cleanup_vggt()
        logger.debug("VGGT resources cleaned up after export")