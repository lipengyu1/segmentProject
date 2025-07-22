import cv2

# 存储点击坐标的列表
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 添加坐标到列表，格式为 [x, y]
        points.append([x, y])
        print(f"Clicked at: [{x}, {y}]")
        # 在点击位置绘制蓝色圆点
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # BGR 蓝色，半径 5，实心圆
        cv2.imshow("Image", img)  # 更新图像显示

# 加载并调整图像大小
img = cv2.imread("E:\\PythonProject\\vggt\\examples\\strawberry\\images\\3-07.jpg")
# img = cv2.resize(img, (1024, 1024))
# img = cv2.resize(img, (2560, 1920))


# 显示图像并设置鼠标回调
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

# 等待按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印所有点击坐标，格式为 [[x1, y1], [x2, y2], ...]
print("All clicked points:", points)