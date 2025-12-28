CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(raycasting)

SET(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(PCL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
        include
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

LINK_LIBRARIES(
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

#ADD_EXECUTABLE(show src/show.cpp)
#ADD_EXECUTABLE(filter_label src/filter_label.cpp)
#ADD_EXECUTABLE(gen_label src/gen_label.cpp)
ADD_EXECUTABLE(nuScenesCAD_gen_label src/nuScenesCAD_gen_label.cpp)
