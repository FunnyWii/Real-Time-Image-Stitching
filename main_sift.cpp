// @https://blog.csdn.net/dcrmg/article/details/52629856

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
 
using namespace cv;
using namespace std;
 
//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri);
 
int main(int argc,char *argv[])  
{  
	Mat image01=imread(argv[1]);  
	Mat image02=imread(argv[2]);
	imshow("拼接图像1",image01);
	imshow("拼接图像2",image02);
 
	//灰度图转换
	Mat image1,image2;  
	cvtColor(image01,image1,cv::COLOR_BGR2GRAY);
	cvtColor(image02,image2,cv::COLOR_BGR2GRAY);
 
	//提取特征点  
	Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(800); // 海塞矩阵阈值
    vector<KeyPoint> keyPoint1, keyPoint2;
    sift->detect(image1, keyPoint1);
    sift->detect(image2, keyPoint2);
 
	//特征点描述，为下边的特征点匹配做准备  
	// SiftDescriptorExtractor siftDescriptor;  
	Mat imageDesc1,imageDesc2;  
	sift->compute(image1, keyPoint1, imageDesc1);
    sift->compute(image2, keyPoint2, imageDesc2);
 
	//获得匹配特征点，并提取最优配对  	
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;  
	matcher.match(imageDesc1,imageDesc2,matchePoints,Mat());
	sort(matchePoints.begin(),matchePoints.end()); //特征点排序	
	//获取排在前N个的最优匹配特征点
	vector<Point2f> imagePoints1,imagePoints2;
	for(int i=0;i<10;i++)
	{		
		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);		
		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);		
	}
 
	//获取图像1到图像2的投影映射矩阵，尺寸为3*3
	Mat homo=findHomography(imagePoints1,imagePoints2,cv::RANSAC);	
    std::cout << "Homography Matrix: \n" << homo << std::endl;

	// cv::Mat homo = (cv::Mat_<double>(3, 3) << 
    // 2.197386425140752, -0.07377270704925194, -1167.414033220644,
    // 0.5110324381673975, 1.935452095463473, -373.3411584695648,
    // 0.001217054710117502, 7.662880411856859e-05, 1);

	Mat adjustMat=(Mat_<double>(3,3)<<1.0,0,image01.cols,0,1.0,0,0,0,1.0);
	Mat adjustHomo=adjustMat*homo;
 
	//获取最强配对点在原始图像和矩阵变换后图像上的对应位置，用于图像拼接点的定位
	Point2f originalLinkPoint,targetLinkPoint,basedImagePoint;
	originalLinkPoint=keyPoint1[matchePoints[0].queryIdx].pt;
	targetLinkPoint=getTransformPoint(originalLinkPoint,adjustHomo);
	basedImagePoint=keyPoint2[matchePoints[0].trainIdx].pt;
 
	//图像配准
	Mat imageTransform1;
	warpPerspective(image01,imageTransform1,adjustMat*homo,Size(image02.cols+image01.cols+10,image02.rows));
 
	//在最强匹配点的位置处衔接，最强匹配点左侧是图1，右侧是图2，这样直接替换图像衔接不好，光线有突变
	Mat ROIMat=image02(Rect(Point(basedImagePoint.x,0),Point(image02.cols,image02.rows)));	
	ROIMat.copyTo(Mat(imageTransform1,Rect(targetLinkPoint.x,0,image02.cols-basedImagePoint.x+1,image02.rows)));
 
	namedWindow("拼接结果",0);
	imshow("拼接结果",imageTransform1);	
	imwrite("拼接结果.jpg",imageTransform1);	
	waitKey();  
	return 0;  
}
 
//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f originalPoint,const Mat &transformMaxtri)
{
	Mat originelP,targetP;
	originelP=(Mat_<double>(3,1)<<originalPoint.x,originalPoint.y,1.0);
	targetP=transformMaxtri*originelP;
	float x=targetP.at<double>(0,0)/targetP.at<double>(2,0);
	float y=targetP.at<double>(1,0)/targetP.at<double>(2,0);
	return Point2f(x,y);
}