# CMake 工程，读者应该熟悉了，我就不一一注释了
cmake_minimum_required( VERSION 2.8 )
project( yzy_slam )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( PCL REQUIRED )
find_package( Ceres REQUIRED )


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories( 
${EIGEN3_INCLUDE_DIR} 
${CHOLMOD_INCLUDE_DIR} 
${PCL_INCLUDE_DIR}
"/usr/include/pcl-1.7"
${CERES_INCLUDE_DIRS}
)

add_executable( yzy_slam main.cpp )
target_link_libraries( yzy_slam
    ${OpenCV_LIBS}
${PCL_COMMON_LIBRARIES} 
${PCL_IO_LIBRARIES}
${PCL_VISUALIZATION_LIBRARIES}
${CERES_LIBRARIES}

    )
