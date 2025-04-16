cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(StitchTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
set(OpenCV_DIR "/home/jetson/opencv-4.5.4/build")
# set(OpenCV_DIR "/home/zch/Downloads/opencv-4.2.0/build")
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(StitchTest main.cpp)
# add_executable(StitchTestORB main_orb.cpp)
# add_executable(StitchTestSift main_sift.cpp)
add_executable(StitchTestDetail main_detail454.cpp)
# add_executable(StitchTestStitcher main_stitcher.cpp)
add_executable(time timeTest.cpp)

# 链接OpenCV库
target_link_libraries(StitchTest ${OpenCV_LIBS})
# target_link_libraries(StitchTestORB ${OpenCV_LIBS})
# target_link_libraries(StitchTestSift ${OpenCV_LIBS})
target_link_libraries(StitchTestDetail ${OpenCV_LIBS})
# target_link_libraries(StitchTestStitcher ${OpenCV_LIBS})
target_link_libraries(time ${OpenCV_LIBS})