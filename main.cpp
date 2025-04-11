#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <chrono>

void OptimizeSeam(cv::Mat& img1, cv::Mat& trans, cv::Mat& dst);

struct four_corners_t
{ 
    cv::Point2f left_top{0.0f, 0.0f};
    cv::Point2f left_bottom{0.0f, 0.0f};
    cv::Point2f right_top{0.0f, 0.0f};
    cv::Point2f right_bottom{0.0f, 0.0f};
};
four_corners_t corners;

void CalculateCorners(const cv::Mat& H, const cv::Mat& src){
    double v2[] = { 0, 0, 1 }; // 左上角
    double v1[3]; // 变换后的坐标值
    cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);  // 列向量
    cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);  // 列向量

    V1 = H * V2;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    // 左下角(0, src.rows, 1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    // 右上角(src.cols, 0, 1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    // 右下角(src.cols, src.rows, 1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2);
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];
}


void OptimizeHomo(cv::Mat img, cv::Mat H, cv::Mat& optimizedH, int& new_width, int& new_height){
    // 计算图像的四个角点
    std::vector<cv::Point2f> corners = { {0, 0}, {img.cols, 0}, {0, img.rows}, {img.cols, img.rows} };
    std::vector<cv::Point2f> transformedCorners;
    cv::perspectiveTransform(corners, transformedCorners, H);

    // 找到变换后角点的最小和最大坐标
    float min_x = std::min({transformedCorners[0].x, transformedCorners[1].x, transformedCorners[2].x, transformedCorners[3].x});
    float min_y = std::min({transformedCorners[0].y, transformedCorners[1].y, transformedCorners[2].y, transformedCorners[3].y});
    float max_x = std::max({transformedCorners[0].x, transformedCorners[1].x, transformedCorners[2].x, transformedCorners[3].x});
    float max_y = std::max({transformedCorners[0].y, transformedCorners[1].y, transformedCorners[2].y, transformedCorners[3].y});

    // 计算新的输出图像尺寸
    new_width = static_cast<int>(max_x - min_x);
    new_height = static_cast<int>(max_y - min_y);

    // 平移矩阵，用于将图像平移到新的坐标系中
    cv::Mat translationMat = (cv::Mat_<double>(3, 3) << 
        1, 0, -min_x,
        0, 1, -min_y,
        0, 0, 1);

    optimizedH = translationMat * H;
    std::cout << "new_width: " << new_width << std::endl;
    std::cout << "new_height: " << new_height << std::endl;
    std::cout << "Optimized Homography Matrix: \n" << optimizedH << std::endl;
    
}
int main(){
    std::string tempImagePath = "../TempImg/";
    cv::Mat imgL, imgM, imgR;
    imgL = cv::imread("../imgs/left.jpg");
    imgM = cv::imread("../imgs/right.jpg");
    imgR = cv::imread("../imgs/r20.jpg");
    if (imgL.empty() || imgM.empty() || imgR.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }
    cv::imshow("imgL", imgL);
    cv::imshow("imgM", imgM);
    // cv::imshow("imgR", imgR);

    // TODO: Add gray image channel judge
    cv::Mat grayL, grayM, grayR;
    cv::cvtColor(imgL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgM, grayM, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(imgR, grayR, cv::COLOR_BGR2GRAY);
    cv::imshow("grayL", grayL);
    cv::imshow("grayM", grayM);
    // cv::imshow("grayR", grayR);
    
    // Extract keypoints
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypointsL, keypointsM, keypointsR;
    // orb->detect(grayL, keypointsL);
    // orb->detect(grayM, keypointsM);
    // orb->detect(grayR, keypointsR);

    // Calculate descriptors for keypoints
    cv::Mat descriptorsL, descriptorsM, descriptorsR;
    sift->detectAndCompute(grayL, cv::noArray(), keypointsL, descriptorsL);
    sift->detectAndCompute(grayM, cv::noArray(), keypointsM, descriptorsM);
    // orb->compute(grayR, keypointsR, descriptorsR);


    //Debug, Show the image with keypoints
    cv::Mat imgWithKeypointsL, imgWithKeypointsM, imgWithKeypointsR;
    cv::drawKeypoints(imgL, keypointsL, imgWithKeypointsL);
    cv::drawKeypoints(imgM, keypointsM, imgWithKeypointsM);
    // cv::drawKeypoints(imgR, keypointsR, imgWithKeypointsR);
    cv::imshow("imgWithKeypointsL", imgWithKeypointsL);
    cv::imshow("imgWithKeypointsM", imgWithKeypointsM);
    // cv::imshow("imgWithKeypointsR", imgWithKeypointsR);
    cv::imwrite(tempImagePath + "/imgWithKeypointsL.jpg", imgWithKeypointsL);
    cv::imwrite(tempImagePath + "/imgWithKeypointsM.jpg", imgWithKeypointsM);
    // cv::imwrite(tempImagePath + "/imgWithKeypointsR.jpg", imgWithKeypointsR);


    // Debug, Show the descriptors(Failed)
    // cv::Mat descriptorsVis;
    // cv::normalize(descriptorsL, descriptorsVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("Descriptors Visualization", descriptorsVis);

    // Matcher
    /*
    TODO: 
        Test Brute-Force Matcher (BFMatcher) -> testing
        Test FLANN-Based Matcher
        Test KNN Matching
        Test Cross-Check Matching
    */
    cv::BFMatcher bf(cv::NORM_HAMMING, true); // 第二个参数为 true 表示使用交叉检查
    cv::FlannBasedMatcher flann;
    std::vector<cv::DMatch> firstMatches;
    flann.match(descriptorsL, descriptorsM, firstMatches);

    // Sort matches by distance, method 1
    std::vector<cv::DMatch> goodMatches;
    // double matchThres = 0.7;        // Need to be tuned
    // for(int i = 0; i  < firstMatches.size(); ++i){
    //     if(firstMatches[i].distance < matchThres * firstMatches[i + 1].distance){
    //         goodMatches.push_back(firstMatches[i]);
    //     }
    // }  
    // Sort matches by distance, method 2
    double max_dist = 0, min_dist = 100;
    for (int i = 0; i < descriptorsL.rows; i++) {
        double dist = firstMatches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::cout << "Max dist: " << max_dist << std::endl;
    std::cout << "Min dist: " << min_dist << std::endl;
    for (int i = 0; i < firstMatches.size(); i++) {
        if (firstMatches[i].distance < 1.3 * min_dist) {
            goodMatches.push_back(firstMatches[i]);
        }
    }

    // Debug, draw the final match results
    cv::Mat matchesForDraw;
    cv::drawMatches(imgL, keypointsL, imgM, keypointsM, goodMatches, matchesForDraw);
    cv::imshow("Final Match ", matchesForDraw);
    cv::imwrite(tempImagePath + "/FinalMatch.jpg", matchesForDraw);

    std::vector<cv::Point2f> imagePointsL, imagePointsM;
    for(int i = 0; i < goodMatches.size(); ++i){
        imagePointsL.push_back(keypointsL[goodMatches[i].queryIdx].pt);
        imagePointsM.push_back(keypointsM[goodMatches[i].trainIdx].pt);
    }

    // Calculate homography trans matrix
    cv::Mat homographyMat = cv::findHomography(imagePointsL, imagePointsM, cv::RANSAC);
    std::cout << "Homography Matrix: \n" << homographyMat << std::endl;
    cv::Mat optimizedHomo;
    int new_width, new_height;
    OptimizeHomo(imgL, homographyMat, optimizedHomo, new_width, new_height);

    // homographyMat = (cv::Mat_<double>(3, 3) << 
    // 2.197386425140752, -0.07377270704925194, -1167.414033220644,
    // 0.5110324381673975, 1.935452095463473, -373.3411584695648,
    // 0.001217054710117502, 7.662880411856859e-05, 1);
    CalculateCorners(homographyMat, imgL);
    std::cout << "left_top: " << corners.left_top << std::endl;
    std::cout << "left_bottom: " << corners.left_bottom << std::endl;
    std::cout << "right_top: " << corners.right_top << std::endl;
    std::cout << "right_bottom: " << corners.right_bottom << std::endl;

    // Image Registration
    cv::Mat imgTransformL;
    // cv::warpPerspective(imgL, imgTransformL, homographyMat, imgL.size());
    cv::warpPerspective(imgL, imgTransformL, homographyMat, cv::Size(MAX(corners.right_top.x, corners.right_bottom.x), imgM.rows));
    cv::imshow("Image Transform L", imgTransformL);


    //创建拼接后的图,需提前计算图的大小
    int dst_width = 2 * imgM.cols;  //取最右点的长度为拼接图的长度
    int dst_height = imgM.rows;

    cv::Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);
    std::cout << "imgTransformL.cols: " << imgTransformL.cols << std::endl;
    std::cout << "imgTransformL.rows: " << imgTransformL.rows << std::endl;
    imgTransformL.copyTo(dst(cv::Rect(0, 0, imgTransformL.cols, imgTransformL.rows)));
    imgM.copyTo(dst(cv::Rect(imgTransformL.cols, 0, imgM.cols, imgM.rows)));

    imshow("b_dst", dst);




    std::cout << "Calculation Done... " << std::endl;
    int key;
    if(key == 27){
        cv::destroyAllWindows();
    }
    else{
        cv::waitKey(0);
    }
    return 0;
}