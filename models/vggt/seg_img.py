import numpy as np
import cv2
import os
import torch
from segment_anything import SamPredictor, sam_model_registry


def validate_file(file_path, file_type="file"):
    """验证文件是否存在且可访问"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type} 不存在于路径：{file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"无权限读取 {file_type}：{file_path}")


def validate_points(points, width, height, point_type="提示点"):
    """验证提示点是否在图像范围内"""
    if points is not None:
        if not isinstance(points, np.ndarray) or points.shape[1] != 2:
            raise ValueError(f"{point_type} 必须是形状为 [N, 2] 的 NumPy 数组")
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            raise ValueError(f"{point_type} 必须在图像范围内 [0, {width}]x[0, {height}]")


def extract_and_resize_object(mask, image_rgb, output_path, target_size=(1024, 768)):
    """
    从掩码中提取目标物体的最小外接矩形，裁剪并缩放到目标尺寸，添加白色背景。

    参数：
        mask (np.ndarray): 二值化掩码，形状为 [H, W]
        image_rgb (np.ndarray): 原始 RGB 图像，形状为 [H, W, 3]
        output_path (str): 输出图像保存路径
        target_size (tuple): 目标尺寸 (width, height)，默认 (1024, 768)
    """
    # 确保掩码是二值化的
    mask_binary = mask.astype(np.uint8) * 255
    # 查找轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("掩码中未找到任何轮廓")

    # 计算所有轮廓的最小外接矩形
    x_min, y_min = image_rgb.shape[1], image_rgb.shape[0]
    x_max, y_max = 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # 裁剪目标区域
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("无效的最小外接矩形")
    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    cropped_height, cropped_width = cropped_image.shape[:2]
    print(f"裁剪区域尺寸: {cropped_width}x{cropped_height}")

    # 计算缩放比例，保持宽高比
    target_width, target_height = target_size
    img_ratio = cropped_width / cropped_height
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * img_ratio)

    # 缩放裁剪图像
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    print(f"缩放后尺寸: {new_width}x{new_height}")

    # 创建白色背景
    white_background = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    # 计算粘贴位置（居中）
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    # 将缩放后的图像粘贴到白色背景
    white_background[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_image

    # 转换为 BGR 保存
    white_background_bgr = cv2.cvtColor(white_background, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, white_background_bgr)
    print(f"已保存裁剪并缩放的图像到 {output_path}")


def generate_masked_image(checkpoint, image_path, input_points_positive, input_points_negative=None,
                          output_dir="output", target_size=(518, 392)):
    """
    生成目标物体为原始 RGB 颜色、背景为纯白的掩码图像，并保存恢复到原始尺寸的带提示点可视化图像。
    支持正向和负向提示点。用户选择掩码后，提取目标物体并缩放到目标尺寸。

    参数：
        checkpoint (str): SAM 模型检查点路径
        image_path (str): 输入图像路径
        input_points_positive (np.ndarray): 原始图像坐标系中的正向提示点，形状为 [N, 2]
        input_points_negative (np.ndarray, optional): 原始图像坐标系中的负向提示点，形状为 [M, 2]，默认为 None
        output_dir (str): 输出图像保存目录
        target_size (tuple): 裁剪并缩放的目标尺寸，默认为 (1024, 768)
    返回：
        masks: 生成的掩码
        scores: 掩码的置信度分数
        masked_image_resized: 最终的掩码图像（恢复到原始尺寸）
    """
    # 验证文件路径
    validate_file(checkpoint, "检查点文件")
    validate_file(image_path, "图像文件")

    # 检查 GPU 可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确保 GPU 环境配置正确")

    # 加载和预处理图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像：{image_path}")

    original_height, original_width = image.shape[:2]
    print(f"输入图像尺寸: {original_width}x{original_height}")
    validate_points(input_points_positive, original_width, original_height, "正向提示点")
    validate_points(input_points_negative, original_width, original_height, "负向提示点")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 缩放图像到 1024x1024
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image_rgb_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # 缩放提示点到 1024x1024
    scale_x = 1024 / original_width
    scale_y = 1024 / original_height
    scaled_points_positive = input_points_positive * np.array([scale_x, scale_y])
    scaled_points_negative = input_points_negative * np.array(
        [scale_x, scale_y]) if input_points_negative is not None else None

    # 合并正向和负向提示点
    if input_points_negative is not None and len(input_points_negative) > 0:
        scaled_points = np.concatenate([scaled_points_positive, scaled_points_negative], axis=0)
        input_labels = np.concatenate([
            np.ones(len(scaled_points_positive)),  # 正向提示点标签为 1
            np.zeros(len(scaled_points_negative))  # 负向提示点标签为 0
        ])
    else:
        scaled_points = scaled_points_positive
        input_labels = np.ones(len(scaled_points_positive))

    # 初始化 SAM 模型（仿照 train_seg.py 的加载方式）
    model_type = "vit_h"
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to("cuda")
    except Exception as e:
        raise RuntimeError(f"加载 SAM 模型失败：{e}")

    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb_resized)

    # 使用提示点预测掩码
    try:
        masks, scores, _ = predictor.predict(
            point_coords=scaled_points,
            point_labels=input_labels,
            multimask_output=True
        )
    except Exception as e:
        raise RuntimeError(f"预测掩码失败：{e}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path).split('.')[0]

    # 处理并保存每个掩码的结果
    for i, mask in enumerate(masks):
        # 创建纯白背景的输出图像
        white_background = np.ones_like(image_rgb_resized) * 255  # 纯白背景
        masked_image = np.where(mask[..., None], image_rgb_resized, white_background)

        # 恢复到原始尺寸
        masked_image_resized = cv2.resize(
            masked_image,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )
        print(f"masked_image_resized 形状: {masked_image_resized.shape}")

        # 转换回 BGR 格式以保存掩码图像
        masked_image_bgr = cv2.cvtColor(masked_image_resized, cv2.COLOR_RGB2BGR)
        print(f"masked_image_bgr 形状: {masked_image_bgr.shape}")

        # 保存掩码图像
        output_path = os.path.join(output_dir, f"{image_name}_masked_{i}.jpg")
        cv2.imwrite(output_path, masked_image_bgr)
        print(f"已保存掩码图像 {i} 到 {output_path}")

        # 保存带提示点的可视化图像（使用原始尺寸）
        vis_image = masked_image_bgr.copy()  # 复制 BGR 图像
        print(f"vis_image 绘制提示点前的形状: {vis_image.shape}")
        # 绘制正向提示点（红色）
        for point in input_points_positive:
            cv2.circle(
                vis_image,
                (int(point[0]), int(point[1])),
                radius=5,
                color=(0, 0, 255),  # BGR 格式的红色
                thickness=-1  # 填充圆
            )
        # 绘制负向提示点（蓝色）
        if input_points_negative is not None:
            for point in input_points_negative:
                cv2.circle(
                    vis_image,
                    (int(point[0]), int(point[1])),
                    radius=5,
                    color=(255, 0, 0),  # BGR 格式的蓝色
                    thickness=-1  # 填充圆
                )
        vis_image = cv2.resize(vis_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        print(f"vis_image 最终形状: {vis_image.shape}")
        vis_path = os.path.join(output_dir, f"{image_name}_masked_{i}_with_points.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"已保存带提示点的可视化图像到 {vis_path}")

    # 用户选择掩码
    # print("\n生成以下掩码：")
    # for i, score in enumerate(scores):
    #     print(f"掩码 {i}: 分数 = {score:.4f}")
    # while True:
    #     try:
    #         selected_index = int(input("请输入要选择的掩码索引 (0, 1, 2): "))
    #         if 0 <= selected_index < len(masks):
    #             break
    #         print(f"请输入有效的索引 (0 到 {len(masks) - 1})")
    #     except ValueError:
    #         print("请输入一个整数")
    #
    # # 提取选中的掩码并缩放到目标尺寸
    # selected_mask = masks[selected_index]
    # # 缩放掩码到原始尺寸
    # selected_mask_resized = cv2.resize(
    #     selected_mask.astype(np.uint8),
    #     (original_width, original_height),
    #     interpolation=cv2.INTER_NEAREST
    # )
    # output_cropped_path = os.path.join(output_dir, f"{image_name}_cropped_resized.jpg")
    # extract_and_resize_object(selected_mask_resized, masked_image_resized, output_cropped_path, target_size)

    return masks, scores, masked_image_resized


if __name__ == "__main__":
    # 示例用法
    checkpoint = "/home_data/lipy/program/vggt/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"
    image_path = "/home_data/lipy/program/vggt/examples/strawberry/images/3-07.jpg"

    # 0-06
    # input_points_positive = np.array([
    #     [1300, 366], [1344, 533] # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [1850, 497], [1010, 575], [1453, 886] # 负向提示点
    # ])
    # 0-13
    # input_points_positive = np.array([
    #     [1597, 422], [1574, 580], [1401, 945] # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [992, 588], [2117, 723] # 负向提示点
    # ])
    # 1-02
    # input_points_positive = np.array([
    #     [1425, 145], [1443, 424], [1421, 745] # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [992, 635], [1985, 642] # 负向提示点
    # ])
    # 1-30
    # input_points_positive = np.array([
    #     [1254, 480], [1273, 654], [1229, 905] # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [842, 718], [1910, 675] # 负向提示点
    # ])
    # 3-07
    input_points_positive = np.array([
        [1242, 630], [1254, 839] # 正向提示点
    ])
    input_points_negative = np.array([
        [888, 830], [1901, 767]# 负向提示点
    ])


    # 12
    # input_points_positive = np.array([
    #     [434, 154], [433, 220]  # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [170, 315], [770, 224] # 负向提示点
    # ])

    # 07
    # input_points_positive = np.array([
    #     [454, 128], [396, 314]  # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [203, 241], [599, 277] # 负向提示点
    # ])

    # 00
    # input_points_positive = np.array([
    #     [367, 177], [447, 236], [602, 309]  # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [228, 363], [667, 106] # 负向提示点
    # ])

    # 16
    # input_points_positive = np.array([
    #     [406, 160], [269, 189], [523, 174], [515, 45]  # 正向提示点
    # ])
    # input_points_negative = np.array([
    #     [636, 302], [203, 161] # 负向提示点
    # ])

    try:
        masks, scores, output_image = generate_masked_image(
            checkpoint=checkpoint,
            image_path=image_path,
            input_points_positive=input_points_positive,
            input_points_negative=input_points_negative,
            output_dir="/home_data/lipy/program/vggt/datasets/demo002"
        )
    except Exception as e:
        print(f"运行出错: {e}")