cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(PolarCalibration)

# set(OpenCV_DIR "/usr/local/share/OpenCV")
# set(PCL_DIR "/usr/local/share/pcl-1.6")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -unroll-loops -march=native")
# set(CMAKE_LINK_LIBRARY_FLAG "${CMAKE_LINK_LIBRARY_FLAG} -fopenmp")

# set (CMAKE_MODULE_PATH "/usr/local/share/pcl-1.6" ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(OMPL    REQUIRED )
# find_package(Boost REQUIRED)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS filesystem system)
# find_package(Boost 1.49.0)
# find_package(CUDA 5.0 REQUIRED)
find_package(OpenCV 2.4.9 REQUIRED HINTS /home/riccardo/opencv/opencv-2.4.13/install/share)

#include eigen
if( EIGEN3_INCLUDE_DIR )
    # in cache already
    set( EIGEN3_FOUND TRUE )
else (EIGEN3_INCLUDE_DIR)
    find_path( EIGEN3_INCLUDE_DIR NAMES Eigen/Core
        PATH_SUFFIXES eigen3/
        HINTS
        ${INCLUDE_INSTALL_DIR}
        /usr/local/include
        ${KDE4_INCLUDE_DIR}
        )
    include( FindPackageHandleStandardArgs )
    find_package_handle_standard_args( Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR )
    mark_as_advanced( EIGEN3_INCLUDE_DIR )
endif(EIGEN3_INCLUDE_DIR)
include_directories( ${EIGEN3_INCLUDE_DIR} )

#include ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# find_package(OpenCV 3.2.0 REQUIRED)
message(${OpenCV_INCLUDE_DIRS})
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
# find_package(OpenMP REQUIRED)
# find_package(OpenGL REQUIRED)
# find_package(GLUT REQUIRED)
# find_package( Eigen3    REQUIRED )
find_package(PCL REQUIRED)
# find_package(PCL 1.6 REQUIRED)
# find_package(CGAL REQUIRED)
# find_package(GMP REQUIRED)
# find_package(MPFR REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# add_subdirectory(tutorials)
add_subdirectory(src)

add_definitions(-std=c++11)
