// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <chrono>

// int main() {
//     auto start = std::chrono::high_resolution_clock::now();

//     // 读取图像文件
//     cv::Mat image = cv::imread("/home/jetson/Documents/wuba/stitch/Real-Time-Image-Stitching/imgs/l10.jpg");

//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//     // 检查图像是否成功读取
//     if (image.empty()) {
//         std::cout << "无法读取图像！" << std::endl;
//         return -1;
//     }

//     std::cout << "imread 耗时: " << duration << " ms" << std::endl;

//     // 创建一个窗口并显示图像
//     cv::namedWindow("显示图像", cv::WINDOW_AUTOSIZE);
//     cv::imshow("显示图像", image);

//     // 等待按键事件
//     cv::waitKey(0);

//     // 关闭所有窗口
//     cv::destroyAllWindows();

//     return 0;
// }    

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 打开默认相机（通常是 0）
    cv::VideoCapture cap("/dev/video0");

    // 检查相机是否成功打开
    if (!cap.isOpened()) {
        std::cout << "无法打开相机！" << std::endl;
        return -1;
    }

    // 设置分辨率为 1080p（1920x1080）
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // 设置帧率为 30
    cap.set(cv::CAP_PROP_FPS, 15);

    // 设置解码方式为 MJPG
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // 检查设置是否成功
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    char fourccChars[5];
    fourccChars[0] = static_cast<char>(fourcc & 0xFF);
    fourccChars[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    fourccChars[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    fourccChars[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    fourccChars[4] = '\0';

    std::cout << "实际分辨率: " << width << "x" << height << std::endl;
    std::cout << "实际帧率: " << fps << std::endl;
    std::cout << "实际解码方式: " << fourccChars << std::endl;

    cv::Mat frame;
    while (true) {
        // 从相机中读取一帧图像
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "读取帧耗时: " << duration << " ms" << std::endl;
        // 检查是否成功读取帧
        if (frame.empty()) {
            std::cout << "无法读取帧！" << std::endl;
            break;
        }

        // 创建一个窗口并显示帧
        cv::namedWindow("相机图像", cv::WINDOW_AUTOSIZE);
        cv::imshow("相机图像", frame);

        // 按 'q' 键退出循环
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 释放相机资源
    cap.release();

    // 关闭所有窗口
    cv::destroyAllWindows();

    return 0;
}    