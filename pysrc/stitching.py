import glob
import cv2
import numpy as np
from datetime import datetime

# 标定内参
def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                        cv2.fisheye.CALIB_CHECK_COND + \
                        cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.png')
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                                     cv2.CALIB_CB_FAST_CHECK + 
                                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return DIM, K, D

# 去畸变
def undistort(img_path, K, D, DIM=(1920,1080), scale=0.6, imshow=False):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img


K1 = np.array([
                [ 798.44484745463683, 0., 989.42960623654233], 
                [ 0., 799.72700252540187, 565.07964926863974], 
                [ 0., 0., 1. ]
            ])
D1 = np.array([ -0.038532511927525298, 0.0166159242534921,
       -0.011321467955016224, 0.0018400104393374887 ])

# img1 = undistort('20241213164947.bmp', K1, D1)
# img2 = undistort('20241213165012.bmp', K1, D1)

# cv2.imwrite("img1.png", img1)
# cv2.imwrite("img2.png", img2)


# 图像拼接，采用ORB方法
def ORB_stitch_images(img1_path, img2_path, output_path):
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用ORB检测器检测特征点和描述符
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 按照距离排序匹配
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择前N个匹配点
    good_matches = matches[:50]

    # 获取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # 计算透视变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 使用透视变换进行图像配准
    img1_warped = cv2.warpPerspective(img1, M, (w1 + w2, h1))

    # 将第二张图像粘贴到第一张图像的右侧
    img1_warped[0:h2, 0:w2] = img2

    # 保存结果
    cv2.imwrite(output_path, img1_warped)

# ORB_stitch_images('img1.png', 'img2.png', 'img3.png')


def Stitcher_stitch_images(image_files, output_file):
    # 创建Stitcher对象，使用默认参数
    stitcher = cv2.Stitcher_create()

    # 读取图像文件
    images = [cv2.imread(file) for file in image_files]

    # 执行拼接操作
    status, stitched_image = stitcher.stitch(images)

    # 检查拼接状态
    if status == cv2.Stitcher_OK:
        # 拼接成功，保存结果图像
        cv2.imwrite(output_file, stitched_image)
        print(f"全景拼接成功，已保存为 {output_file}")
    else:
        # 拼接失败，打印错误信息
        print(f"全景拼接失败，错误代码: {status}")

# 使用示例 '20241213164904.bmp', 
# image_files = ['./imgs_stitch/180/c10l.jpg', './imgs_stitch/180/c10m.jpg','./imgs_stitch/180/c10r.jpg']  # 替换为你的图像文件路径
# image_files = ['./imgs_stitch/210/c10l.jpg', './imgs_stitch/210/c10m.jpg','./imgs_stitch/210/c10r.jpg']  # 替换为你的图像文件路径
image_files = ['./imgs/l70.jpg','./imgs/lr0.jpg', './imgs/r70.jpg']  # 替换为你的图像文件路径
# image_files = ['./FOV100/l90.jpg', './FOV100/lr0.jpg','./FOV100/r90.jpg']  # 替换为你的图像文件路径
output_file = '10.jpg'  # 输出文件路径
start_time = datetime.now()
Stitcher_stitch_images(image_files, output_file)
end_time = datetime.now()
execution_time = end_time - start_time
print(f"Execution time: {execution_time.total_seconds()} seconds")

# def stitch_images_sift_flann(img1_path, img2_path, output_path):
#     # 读取图像
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)

#     # 转换为灰度图像
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # 初始化SIFT检测器
#     sift = cv2.SIFT_create()

#     # 检测关键点和计算描述符
#     keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

#     # 设置FLANN匹配器
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     # 进行匹配
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     # 根据Lowe's ratio test筛选好的匹配点
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good_matches.append(m)

#     # 获取匹配点的坐标
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

#     # 计算透视变换矩阵
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#     # 获取图像尺寸
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     # 使用透视变换进行图像配准（将img2变换到img1的坐标系）
#     img2_warped = cv2.warpPerspective(img2, M, (w1 + w2, h1))

#     # 将img1和变换后的img2进行融合（这里简单地将img2粘贴到img1的右侧）
#     # 注意：这里没有进行复杂的图像融合处理，只是简单拼接
#     stitched_image = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
#     stitched_image[:h1, :w1] = img1
#     stitched_image[:h2, w1:w1+w2] = img2_warped[:, :w2]  # 只取变换后图像的有效部分

#     # 保存结果
#     cv2.imwrite(output_path, stitched_image)

# # 使用示例
# stitch_images_sift_flann('20241213164947.bmp', '20241213165012.bmp', 'output_panorama.png')



