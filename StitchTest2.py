import cv2
import numpy as np


def stitch_images_with_homography(image_files, output_file):
    # 读取图像文件
    images = [cv2.imread(file) for file in image_files]
    homographies = []

    for i in range(len(images) - 1):
        # 转换为灰度图像
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        # 使用 SIFT 进行特征提取
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # 使用 FLANN 进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # 应用比值测试来筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # 获取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append(H)

        # 进行透视变换
        h, w = images[i].shape[:2]
        warped_image = cv2.warpPerspective(images[i], H, (images[i + 1].shape[1] + w, images[i + 1].shape[0]))

        # 将第二幅图像复制到拼接后的图像上
        warped_image[0:images[i + 1].shape[0], 0:images[i + 1].shape[1]] = images[i + 1]
        images[i + 1] = warped_image

    # 保存最终拼接结果
    cv2.imwrite(output_file, images[-1])
    print(f"全景拼接成功，已保存为 {output_file}")
    return homographies


if __name__ == "__main__":
    image_files = ['./imgs/l20.jpg', './imgs/lr0.jpg']
    output_file = 'stitched_image.jpg'
    homographies = stitch_images_with_homography(image_files, output_file)
    for i, H in enumerate(homographies):
        print(f"第 {i + 1} 对图像的单应矩阵:")
        print(H)
    