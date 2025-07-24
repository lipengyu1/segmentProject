from flask import Blueprint, request, jsonify, session, render_template
import os
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

# 上传和处理文件夹配置
UPLOAD_FOLDER = 'datasets/'
bp.config = {'UPLOAD_FOLDER': UPLOAD_FOLDER}

# 确保基础目录存在
os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'videos'), exist_ok=True)

@bp.route('/', methods=['GET', 'POST'])
def setProjectName():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        project_name = request.form.get('project_name')
        if not project_name:
            logger.error("Project name is empty")
            return jsonify({'error': '项目名不能为空'}), 400

        project_name = ''.join(c for c in project_name if c.isalnum() or c in ['_', '-'])
        if not project_name:
            logger.error("Project name contains invalid characters")
            return jsonify({'error': '项目名包含非法字符'}), 400

        project_path = os.path.join(bp.config['UPLOAD_FOLDER'], project_name, 'images')
        video_path = os.path.join(bp.config['UPLOAD_FOLDER'], project_name, 'videos')
        try:
            os.makedirs(project_path, exist_ok=True)
            os.makedirs(video_path, exist_ok=True)
            logger.debug(f"Created project folders: {project_path}, {video_path}")
        except Exception as e:
            logger.error(f"Failed to create project folders: {str(e)}")
            return jsonify({'error': f'创建项目文件夹失败: {str(e)}'}), 500

        session['project_name'] = project_name
        return jsonify({'message': f'项目 {project_name} 创建成功', 'project_name': project_name})

@bp.route('/upload', methods=['POST'])
def upload_image():
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    project_path = os.path.join(bp.config['UPLOAD_FOLDER'], project_name, 'images')
    if not os.path.exists(project_path):
        logger.error(f"Project folder not found: {project_path}")
        return jsonify({'error': '项目文件夹不存在'}), 400

    files = request.files.getlist('images')  # 改为 'images' 匹配前端
    if not files or all(file.filename == '' for file in files):
        logger.error("No valid files uploaded")
        return jsonify({'error': '没有文件上传'}), 400

    successful_uploads = []
    errors = []

    for file in files:
        if file.filename == '':
            errors.append({'filename': '', 'error': '未选择文件'})
            continue

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = file.filename
            image_path = os.path.join(project_path, filename)
            try:
                file.save(image_path)
                successful_uploads.append({'filename': filename, 'path': image_path})
                logger.debug(f"Saved image: {image_path}")
            except Exception as e:
                errors.append({'filename': filename, 'error': str(e)})
                logger.error(f"Failed to save image {filename}: {str(e)}")
        else:
            errors.append({'filename': file.filename, 'error': '不支持的文件格式'})
            logger.warning(f"Unsupported file format: {file.filename}")

    if successful_uploads and not errors:
        logger.debug(f"Uploaded {len(successful_uploads)} images to {project_path}")
        return jsonify({'message': f'成功上传 {len(successful_uploads)} 张图像', 'uploads': successful_uploads})
    elif successful_uploads:
        logger.debug(f"Partially uploaded {len(successful_uploads)} images with {len(errors)} errors")
        return jsonify({'message': '部分图像上传成功', 'uploads': successful_uploads, 'errors': errors})
    else:
        logger.error(f"All uploads failed: {errors}")
        return jsonify({'error': '所有上传失败', 'errors': errors}), 400

@bp.route('/uploadVideo', methods=['POST'])
def upload_video():
    project_name = session.get('project_name')
    if not project_name:
        logger.error("No project name in session")
        return jsonify({'error': '请先设置项目名'}), 400

    project_path = os.path.join(bp.config['UPLOAD_FOLDER'], project_name, 'videos')
    try:
        os.makedirs(project_path, exist_ok=True)
        logger.debug(f"Ensured video folder exists: {project_path}")
    except Exception as e:
        logger.error(f"Failed to create video folder: {str(e)}")
        return jsonify({'error': f'创建视频文件夹失败: {str(e)}'}), 500

    files = request.files.getlist('videos')  # 改为 'videos' 匹配前端
    if not files or all(file.filename == '' for file in files):
        logger.error("No valid files uploaded")
        return jsonify({'error': '没有视频文件上传'}), 400

    successful_uploads = []
    errors = []

    for file in files:
        if file.filename == '':
            errors.append({'filename': '', 'error': '未选择视频文件'})
            continue

        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            filename = file.filename
            video_path = os.path.join(project_path, filename)
            try:
                file.save(video_path)
                successful_uploads.append({'filename': filename, 'path': video_path})
                logger.debug(f"Saved video: {video_path}")
            except Exception as e:
                errors.append({'filename': filename, 'error': str(e)})
                logger.error(f"Failed to save video {filename}: {str(e)}")
        else:
            errors.append({'filename': file.filename, 'error': '不支持的视频格式'})
            logger.warning(f"Unsupported video format: {file.filename}")

    if successful_uploads and not errors:
        logger.debug(f"Uploaded {len(successful_uploads)} videos to {project_path}")
        return jsonify({'message': f'成功上传 {len(successful_uploads)} 个视频', 'uploads': successful_uploads})
    elif successful_uploads:
        logger.debug(f"Partially uploaded {len(successful_uploads)} videos with {len(errors)} errors")
        return jsonify({'message': '部分视频上传成功', 'uploads': successful_uploads, 'errors': errors})
    else:
        logger.error(f"All video uploads failed: {errors}")
        return jsonify({'error': '所有视频上传失败', 'errors': errors}), 400