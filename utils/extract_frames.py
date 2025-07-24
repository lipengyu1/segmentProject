import os
import cv2
import logging

# 配置日志
logger = logging.getLogger(__name__)


def extract_frames(video_path, images_path, frame_count):
    """
    从视频文件夹中抽取指定数量的帧并保存到图像文件夹。

    参数:
        video_path (str): 视频文件夹路径
        images_path (str): 图像存储文件夹路径
        frame_count (int): 每视频抽取的帧数

    返回:
        tuple: (successful_extractions, errors)
            - successful_extractions: 成功抽取的帧信息列表
            - errors: 错误信息列表
    """
    # 获取视频文件列表
    video_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        logger.error("视频文件夹中没有视频文件")
        return [], [{'filename': '', 'error': '视频文件夹中没有视频文件'}]

    successful_extractions = []
    errors = []

    # 处理每个视频文件
    for video_file in video_files:
        video_filepath = os.path.join(video_path, video_file)
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_filepath)
            if not cap.isOpened():
                errors.append({'filename': video_file, 'error': '无法打开视频文件'})
                logger.error(f"无法打开视频文件: {video_file}")
                continue

            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                errors.append({'filename': video_file, 'error': '视频文件为空'})
                logger.error(f"视频文件为空: {video_file}")
                cap.release()
                continue

            # 计算抽帧间隔
            interval = max(1, total_frames // frame_count)
            extracted_frames = []
            frame_index = 0
            count = 0

            # 抽取指定数量的帧
            while cap.isOpened() and count < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break

                # 保存帧
                frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{count + 1}.jpg"
                frame_filepath = os.path.join(images_path, frame_filename)
                try:
                    cv2.imwrite(frame_filepath, frame)
                    extracted_frames.append({'filename': frame_filename, 'path': frame_filepath})
                    logger.debug(f"保存帧: {frame_filepath}")
                    count += 1
                except Exception as e:
                    errors.append({'filename': frame_filename, 'error': f'保存帧失败: {str(e)}'})
                    logger.error(f"保存帧 {frame_filename} 失败: {str(e)}")
                    break

                frame_index += interval

            cap.release()
            if extracted_frames:
                successful_extractions.append({'video': video_file, 'frames': extracted_frames})
                logger.debug(f"成功从 {video_file} 抽取 {len(extracted_frames)} 帧")

        except Exception as e:
            errors.append({'filename': video_file, 'error': str(e)})
            logger.error(f"处理视频 {video_file} 失败: {str(e)}")

    return successful_extractions, errors