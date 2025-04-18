#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"

#include <iostream>

#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#define DEBUG_PRINT_ERR(x) (std::cerr << x << std::endl)
#define DEBUG_PRINT_OUT(x) (std::cout << x << std::endl)
#else
#define DEBUG_PRINT_ERR(x)
#define DEBUG_PRINT_OUT(x)
#endif

using namespace std;
using namespace cv;

bool try_gpu = false;
float imshow_scale_factor = 0.2f;

int main()
{
    Mat im1 = imread("../imgs/l70.jpg");
    Mat im2 = imread("../imgs/lr0.jpg");
    Mat im3 = imread("../imgs/r70.jpg");

    vector<Mat> imgs;
    vector<Mat> H;
    imgs.push_back(im1);
    imgs.push_back(im2);
    imgs.push_back(im3);

    Mat output;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    // stitcher->setRegistrationResol(0.5);
    // stitcher->setSeamEstimationResol(0.1);
    // stitcher->setCompositingResol(Stitcher::ORIG_RESOL);
    // stitcher->setPanoConfidenceThresh(1.0);
    // stitcher->setWaveCorrection(true);
    // stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
    // Ptr<ORB> orb = ORB::create(3000);
    // stitcher->setFeaturesFinder(orb);
    // stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(try_gpu));
    // stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
    // stitcher->setWarper(makePtr<SphericalWarper>());
    // stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    // stitcher->setSeamFinder(makePtr<detail::VoronoiSeamFinder>());
    // stitcher->setBlender(makePtr<detail::MultiBandBlender>());
    
    Stitcher::Status status = stitcher->estimateTransform(imgs);
    vector<detail::CameraParams> cameras = stitcher->cameras();
    for(int i = 0; i < imgs.size(); i++)
    {
        DEBUG_PRINT_OUT("----Image index " << i << " camera parameters----");
        DEBUG_PRINT_OUT("Camera intrinsic matrix");
        DEBUG_PRINT_OUT(cameras[i].K());

        DEBUG_PRINT_OUT("Focal length : " << cameras[i].focal);

        DEBUG_PRINT_OUT("Aspect ratio : " << cameras[i].aspect);

        DEBUG_PRINT_OUT("Principle Point X : " << cameras[i].ppx);

        DEBUG_PRINT_OUT("Principle Point Y : " << cameras[i].ppy);

        DEBUG_PRINT_OUT("Rotation matrix");
        DEBUG_PRINT_OUT(cameras[i].R);

        DEBUG_PRINT_OUT("Translation matrix");
        DEBUG_PRINT_OUT(cameras[i].t);
        DEBUG_PRINT_OUT(" ");
        cv::Mat K, R, K_inv;
        K = cameras[i].K();
        R = cameras[i].R;
        K_inv = K.inv();
        K.convertTo(K, CV_64F); // 转换为 double 类型
        R.convertTo(R, CV_64F); // 转换为 double 类型
        K_inv.convertTo(K_inv, CV_64F); // 转换为 double 类型

        cv::Mat temp;
        temp = K * R * K_inv;
        H.push_back(temp);
        DEBUG_PRINT_OUT("Homography matrix: ");
        DEBUG_PRINT_OUT(H[i]);

    }
    cv::Mat imageTransform1, imageTransform2, imageTransform3;
    cv::Mat resize1, resize2, resize3;
    // cv::resize(im1,resize1)
    warpPerspective(im1, imageTransform1, H[0], im1.size());
    imshow("imageTransform1 image", imageTransform1);


    status = stitcher->stitch(imgs, output);
    if (status != Stitcher::OK)
    {
        DEBUG_PRINT_ERR("Fail stitching: ");
        switch(status)
        {
            case Stitcher::ERR_NEED_MORE_IMGS :
                DEBUG_PRINT_ERR("Need more images");
                break;
            case Stitcher::ERR_HOMOGRAPHY_EST_FAIL :
                DEBUG_PRINT_ERR("Homography estimation failed");
                break;
            case Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL :
                DEBUG_PRINT_ERR("Camera parameter adjustment failed");
                break;
            default :
                DEBUG_PRINT_ERR("Unknown Error");
        }
    }
    else
    {
        DEBUG_PRINT_OUT("resize output panorama image and show it");
        Mat tmp;
        // resize(output, tmp, Size(), imshow_scale_factor, imshow_scale_factor);
        imshow("Panorama image", output);
        imwrite("Panoramaimage.jpg", output);
        waitKey(0);
    }

    

    return 0;
}