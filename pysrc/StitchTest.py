import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    # 检查是否成功检测到特征点和描述符
    if key_points1 is None or key_points2 is None or descriptor1 is None or descriptor2 is None:
        raise ValueError("Failed to detect keypoints or descriptors in one or both images.")
    else:
        print(f"Detected {len(key_points1)} keypoints in left image and {len(key_points2)} keypoints in right image.")
    # 检查描述符是否为空
    if len(descriptor1) == 0 or len(descriptor2) == 0:
        raise ValueError("Descriptors are empty. Unable to proceed with matching.")
    else:
        print(f"Descriptor lengths: left image: {len(descriptor1)}, right image: {len(descriptor2)}")
    
    good_matches = match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
    # 检查是否找到足够的匹配点
    print(f"Number of good matches: {len(good_matches)}")
    if len(good_matches) < 4:  # 至少需要 4 个点来计算单应矩阵
        raise ValueError(f"Not enough good matches found: {len(good_matches)} (at least 4 required).")
    else:
        print(f"Found {len(good_matches)} good matches.")

    
    final_H = ransac(good_matches)
    # 检查是否成功计算单应矩阵
    if final_H is None or final_H.size == 0:
        raise ValueError("Failed to compute the homography matrix.")
    else:
        print("Homography matrix calculated successfully.")
        print(final_H)
    
    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 =  cv2.perspectiveTransform(points, final_H)
    list_of_points = np.concatenate((points1,points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max-x_min, y_max-y_min))
    output_img[(-y_min):rows1+(-y_min), (-x_min):cols1+(-x_min)] = right_img
    result_img = output_img
    return result_img
    
    # TO DO: implement your solution here
    # raise NotImplementedError
    
def get_keypoint(left_img, right_img):
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()

    # key_points1 = sift.detect(l_img, None)
    # key_points1, descriptor1 = surf.compute(l_img, key_points1)
    
    # key_points2 = sift.detect(r_img, None)
    # key_points2, descriptor2 = surf.compute(r_img, key_points2)
    key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(r_img, None)
    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    #k-Nearest neighbours between each descriptor
    i = 0
    k=2
    all_matches = []
    for d1 in descriptor1:
      dist = []
      j = 0
      for d2 in descriptor2:
          dist.append([i, j, np.linalg.norm(d1 - d2)])
          j = j + 1
      dist.sort(key=lambda x: x[2])
      all_matches.append(dist[0:k])
      i = i + 1

    # Ratio test to get good matches
    good_matches = []
    for m, n in all_matches:
        if m[2] < 0.75*n[2]:
            left_pt = key_points1[m[0]].pt
            right_pt = key_points2[m[1]].pt
            good_matches.append(
                [left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

def homography(points):
    A = []
    for pt in points:
      x, y = pt[0], pt[1]
      X, Y = pt[2], pt[3]
      A.append([x, y, 1, 0, 0, 0, -1 * X * x, -1 * X * y, -1 * X])
      A.append([0, 0, 0, x, y, 1, -1 * Y * x, -1 * Y * y, -1 * Y])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = (vh[-1, :].reshape(3, 3))
    H = H/ H[2, 2]
    return H

def ransac(good_pts):
    best_inliers = []
    final_H = []
    t=5
    for i in range(5000):
        random_pts = random.choices(good_pts, k=4)
        H = homography(random_pts)
        inliers = []
        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp = Hp / Hp[2]
            dist = np.linalg.norm(p_1 - Hp)

            if dist < t: inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers,final_H = inliers,H
    return final_H

if __name__ == "__main__":
    left_img = cv2.imread('./imgs/l20.jpg')
    right_img = cv2.imread('./imgs/lr0.jpg')

    if left_img is None or right_img is None:
        raise FileNotFoundError("One or both images could not be loaded. Please check the file paths.")
    
    result_img = solution(left_img, right_img)
    cv2.imwrite('33.jpg', result_img)