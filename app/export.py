from flask import Blueprint, request, jsonify, session, send_file
import os
import logging
import threading
import zipfile
import io
import shutil
import tempfile

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

        # 创建包含 images 和 sparse 文件夹的 zip 文件
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"{project_name}_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加 images 文件夹
            for root, _, files in os.walk(image_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join('datasets', project_name))
                    zipf.write(file_path, os.path.join(project_name, arcname))
            # 添加 sparse 文件夹
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join('datasets', project_name))
                    zipf.write(file_path, os.path.join(project_name, arcname))

        logger.debug(f"Zip file created at {zip_path}")
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{project_name}_export.zip"
        )
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'导出失败: {str(e)}'}), 500
    finally:
        # 清理 VGGT 资源
        from utils.reconstruct_images import cleanup_vggt
        cleanup_vggt()
        logger.debug("VGGT resources cleaned up after export")
        # 清理临时文件
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Temporary files cleaned up")