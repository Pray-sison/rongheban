#cmake_minimum_required(VERSION 3.4.1)
#
#project(libtrack_core.so.0.1.0)
#
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
## 设置链接器标志以允许未定义的共享库
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
#
#set(RKNN_DIR ${CMAKE_SOURCE_DIR}/dependents/rknn)
#set(RKNN_INCLUDE_DIRS ${RKNN_DIR}/include)
#set(RKNN_LIBS_DIR ${RKNN_DIR}/lib)
#
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/dependents/opencv-linux-aarch64)
#set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)
#
#set(OpenCV_LIBS_DIR ${OpenCV_DIR}/lib)
#
#set(Eigen_DIR ${CMAKE_SOURCE_DIR}/dependents/eigen)
#set(Eigen_INCLUDE_DIRS ${Eigen_DIR}/include/eigen3)
#
#
#
#include_directories(${RKNN_INCLUDE_DIRS})
#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${Eigen_INCLUDE_DIRS})
#include_directories(${CMAKE_SOURCE_DIR}/yolo/include)
##include_directories(${CMAKE_SOURCE_DIR}/deepsort/include)
#include_directories(${CMAKE_SOURCE_DIR}/praytrack/include)
#include_directories(${CMAKE_SOURCE_DIR}/kcf/include)
#include_directories(${CMAKE_SOURCE_DIR}/interface/include)
#
#file(GLOB YOLO_SOURCES "${CMAKE_SOURCE_DIR}/yolo/src/*.cpp")
##file(GLOB DEEPSORT_SOURCES "${CMAKE_SOURCE_DIR}/deepsort/src/*.cpp")
#file(GLOB PRAYTRACK_SOURCES "${CMAKE_SOURCE_DIR}/praytrack/src/*.cpp")
#file(GLOB KCF_SOURCES "${CMAKE_SOURCE_DIR}/kcf/src/*.cpp")
#file(GLOB INTERFACE_SOURCES "${CMAKE_SOURCE_DIR}/interface/src/*.cpp")
#
## 创建一个共享库
#add_library(libtrack_core.so.0.1.0 SHARED
#        ${YOLO_SOURCES}
#        ${PRAYTRACK_SOURCES}
#        ${KCF_SOURCES}
#        ${INTERFACE_SOURCES}
#        )
#
##file(GLOB OPENCV_LIBRARIES "${OpenCV_LIBS_DIR}/*.so")
##file(GLOB RKNN_LIBRARIES "${RKNN_LIBS_DIR}/*.so")
##
### 链接所需的库
##target_link_libraries(mylibrary
##        ${OPENCV_LIBRARIES}
##        ${RKNN_LIBRARIES}
##        )
#
## 添加头文件目录
#target_include_directories(libtrack_core.so.0.1.0 PRIVATE
#        ${CMAKE_SOURCE_DIR}/yolo/include
#        ${CMAKE_SOURCE_DIR}/praytrack/include
#        ${CMAKE_SOURCE_DIR}/kcf/include
#        )
#
## 添加头文件目录
#target_include_directories(libtrack_core.so.0.1.0 PUBLIC
#        ${CMAKE_SOURCE_DIR}/interface/include
#        )
#
## 修改安装路径，将库和公共头文件安装到 ${CMAKE_SOURCE_DIR}/mylib
#install(TARGETS libtrack_core.so.0.1.0
#        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track/lib
#        ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track/lib
#        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track/bin
#        )
#
## 安装共有头文件
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/interface/include
#        DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track
#        )
#
## 安装私有头文件
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/yolo/include
#        DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track
#        )
#
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/praytrack/include
#        DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track
#        )
#
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/kcf/include
#        DESTINATION ${CMAKE_SOURCE_DIR}/dependents/track
#        )


cmake_minimum_required(VERSION 3.4.1)

project(detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置链接器标志以允许未定义的共享库
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

set(RKNN_DIR ${CMAKE_SOURCE_DIR}/dependents/rknn)
set(RKNN_INCLUDE_DIRS ${RKNN_DIR}/include)
set(RKNN_LIBS_DIR ${RKNN_DIR}/lib)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/dependents/opencv)
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include/opencv4)
set(OpenCV_LIBS_DIR ${OpenCV_DIR}/lib)

#set(My_DIR ${CMAKE_SOURCE_DIR}/dependents/track)
#set(My_INCLUDE_DIRS ${My_DIR}/include)
#set(My_LIBS_DIR ${My_DIR}/lib)

set(Eigen_DIR ${CMAKE_SOURCE_DIR}/dependents/eigen)
set(Eigen_INCLUDE_DIRS ${Eigen_DIR}/include/eigen3)

include_directories(${RKNN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
#include_directories(${My_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/yolo/include)
include_directories(${CMAKE_SOURCE_DIR}/praytrack/include)
#include_directories(${CMAKE_SOURCE_DIR}/deepsort/include)
include_directories(${CMAKE_SOURCE_DIR}/kcf/include)
include_directories(${CMAKE_SOURCE_DIR}/interface/include)


file(GLOB YOLO_SOURCES "${CMAKE_SOURCE_DIR}/yolo/src/*.cpp")
#file(GLOB DEEPSORT_SOURCES "${CMAKE_SOURCE_DIR}/deepsort/src/*.cpp")
file(GLOB PRAYTRACK_SOURCES "${CMAKE_SOURCE_DIR}/praytrack/src/*.cpp")
file(GLOB KCF_SOURCES "${CMAKE_SOURCE_DIR}/kcf/src/*.cpp")
file(GLOB INTERFACE_SOURCES "${CMAKE_SOURCE_DIR}/interface/src/*.cpp")
add_executable(detect ${CMAKE_SOURCE_DIR}/main.cpp ${YOLO_SOURCES} ${PRAYTRACK_SOURCES} ${KCF_SOURCES} ${INTERFACE_SOURCES})
#add_executable(detect ${CMAKE_SOURCE_DIR}/main.cpp)

file(GLOB OPENCV_LIBRARIES "${OpenCV_LIBS_DIR}/*.so")
file(GLOB RKNN_LIBRARIES "${RKNN_LIBS_DIR}/*.so")
#file(GLOB MY_LIBRARIES "${My_LIBS_DIR}/*.so")


target_link_libraries(detect
#        ${MY_LIBRARIES}
        ${OPENCV_LIBRARIES}
        ${RKNN_LIBRARIES}
        )
