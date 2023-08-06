cmake_minimum_required(VERSION 2.8)

PROJECT(plate_rec)


set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lutil")

SET(CMAKE_BUILD_TYPE "Release")# "Release"

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -DGLEW_NO_GLU -std=c++11 -fPIC -fpermissive")
# "${CMAKE_CXX_FLAGS} -W -Wall -DGLEW_NO_GLU -std=c++11 -fPIC -fpermissive"-g2 -ggdb

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include
    #${PROJECT_SOURCE_DIR}/dependencies/GL_include/glog_all
    ${PROJECT_SOURCE_DIR}/dependencies/include
  )

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/dependencies/lib)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/dependencies/lib )

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

ADD_LIBRARY(plate_rec SHARED ${DIR_SRCS})

ADD_EXECUTABLE(${PROJECT_NAME}_test  ${PROJECT_SOURCE_DIR}/test/test_image_vehicle.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test  avcodec avutil swscale avformat plate_rec onnxruntime opencv_freetype opencv_video opencv_videoio opencv_core opencv_highgui  opencv_imgproc opencv_imgcodecs)

