import cv2
import numpy as np

# 定义 Homography Matrix
homography_matrix = np.array([
    [0.3383157774188078, -3.208080683252875, 1483.277395102418],
    [0.105811266883179, -1.10817837048159, 515.8693761058087],
    [0.0001962928960386732, -0.002142453134478259, 0.9999999999999999]
])

# 创建一个 1920x1080 的图像（白色背景）
image = cv2.imread("./imgs/l20.jpg")

# 在图像上绘制一些标记点（例如四个角）
cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)         # 左上角
cv2.circle(image, (1919, 0), 10, (0, 255, 0), -1)      # 右上角
cv2.circle(image, (0, 1079), 10, (255, 0, 0), -1)      # 左下角
cv2.circle(image, (1919, 1079), 10, (0, 255, 255), -1) # 右下角

# 使用 warpPerspective 进行透视变换
transformed_image = cv2.warpPerspective(image, homography_matrix, (1280, 720))

# 显示原始图像和变换后的图像
cv2.imshow("Original Image", image)
cv2.imshow("Transformed Image", transformed_image)

# 保存结果
cv2.imwrite("original_image.jpg", image)
cv2.imwrite("transformed_image.jpg", transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()