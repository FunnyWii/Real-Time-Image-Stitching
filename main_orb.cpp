#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
} four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat& H, const Mat& src)
{
    double v2[] = { 0, 0, 1 }; // 左上角
    double v1[3]; // 变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  // 列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  // 列向量

    V1 = H * V2;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    // 左下角(0, src.rows, 1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    // 右上角(src.cols, 0, 1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    // 右下角(src.cols, src.rows, 1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];
}

int main(int argc, char* argv[])
{
    Mat image01 = imread("../imgs/r10.jpg", IMREAD_COLOR); // 右图
    Mat image02 = imread("../imgs/l10.jpg", IMREAD_COLOR); // 左图
    if (image01.empty() || image02.empty()) {
        cerr << "Error: Could not load images!" << endl;
        return -1;
    }

    imshow("p2", image01);
    imshow("p1", image02);

    // 灰度图转换
    Mat image1, image2;
    cvtColor(image01, image1, COLOR_BGR2GRAY);
    cvtColor(image02, image2, COLOR_BGR2GRAY);

    // 提取特征点
    Ptr<ORB> orb = ORB::create(3000);
    vector<KeyPoint> keyPoint1, keyPoint2;
    orb->detect(image1, keyPoint1);
    orb->detect(image2, keyPoint2);

    // 特征点描述
    Mat imageDesc1, imageDesc2;
    orb->compute(image1, keyPoint1, imageDesc1);
    orb->compute(image2, keyPoint2, imageDesc2);

    // 使用 BFMatcher 替代 flann::Index
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(imageDesc2, imageDesc1, knnMatches, 2);

    // Lowe's algorithm, 获取优秀匹配点
    vector<DMatch> GoodMatchePoints;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < 0.4 * knnMatch[1].distance) {
            GoodMatchePoints.push_back(knnMatch[0]);
        }
    }

    Mat first_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    imshow("first_match", first_match);

    vector<Point2f> imagePoints1, imagePoints2;
    for (const auto& match : GoodMatchePoints) {
        imagePoints2.push_back(keyPoint2[match.queryIdx].pt);
        imagePoints1.push_back(keyPoint1[match.trainIdx].pt);
    }

    // 获取图像1到图像2的投影映射矩阵
    Mat homo = findHomography(imagePoints1, imagePoints2, RANSAC);
    cout << "变换矩阵为：\n" << homo << endl;

    // 计算配准图的四个顶点坐标
    CalcCorners(homo, image01);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    // 图像配准
    Mat imageTransform1;
    warpPerspective(image01, imageTransform1, homo, Size(max(corners.right_top.x, corners.right_bottom.x), image02.rows));
    imshow("直接经过透视矩阵变换", imageTransform1);
    imwrite("trans1.jpg", imageTransform1);

    // 创建拼接后的图
    int dst_width = imageTransform1.cols;
    int dst_height = image02.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));

    imshow("b_dst", dst);

    OptimizeSeam(image02, imageTransform1, dst);

    imshow("dst", dst);
    imwrite("dst.jpg", dst);

    waitKey();

    return 0;
}

// 优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
    int start = min(corners.left_top.x, corners.left_bottom.x); // 开始位置，即重叠区域的左边界

    double processWidth = img1.cols - start; // 重叠区域的宽度
    int rows = dst.rows;
    int cols = img1.cols; // 注意，是列数*通道数
    double alpha = 1; // img1中像素的权重
    for (int i = 0; i < rows; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  // 获取第i行的首地址
        uchar* t = trans.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = start; j < cols; j++)
        {
            // 如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }
            else
            {
                // img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
                alpha = (processWidth - (j - start)) / processWidth;
            }

            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}
