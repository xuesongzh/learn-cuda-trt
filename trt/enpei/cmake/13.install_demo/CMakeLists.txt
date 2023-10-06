cmake_minimum_required(VERSION 3.10)

project(instal_demo)

# 添加头文件
include_directories(include)

# 添加静态库
add_library(slib STATIC src/slib.cpp include/slib.h)
# 添加动态库
add_library(dlib SHARED src/dlib.cpp include/dlib.h)

# 设置RPATH，否则install后，运行时找不到动态库
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 添加可执行文件
add_executable(instal_demo main.cpp)
# 链接库
target_link_libraries(instal_demo slib dlib)

# 设置公共头文件，以便install时，将头文件一起安装，或者使用install(DIRECTORY include/ DESTINATION include)
set_target_properties(slib PROPERTIES PUBLIC_HEADER include/slib.h)
set_target_properties(dlib PROPERTIES PUBLIC_HEADER include/dlib.h)

# 安装头文件
# install(DIRECTORY include/ DESTINATION include)

# 设置安装
install(TARGETS instal_demo slib dlib
        RUNTIME DESTINATION bin # 可执行文件
        LIBRARY DESTINATION lib # 动态库
        ARCHIVE DESTINATION lib # 静态库
        PUBLIC_HEADER DESTINATION include # 公共头文件
        )
        
#[[
如果不设置DCMAKE_INSTALL_PREFIX ，则会安装到 /usr/local 目录下
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./installed
或者其他目录
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/Documents/install_demo
cmake --build build
cmake --install build
]]        

