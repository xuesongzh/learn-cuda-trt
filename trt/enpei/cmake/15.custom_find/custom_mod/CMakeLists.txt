# 最低版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(custom_mod)

# 添加动态库
add_library(dlib SHARED src/dlib.cpp)
# 添加头文件目录
target_include_directories(dlib PUBLIC include)

# 设置公共头文件
set_target_properties(dlib PROPERTIES PUBLIC_HEADER include/dlib.h)

# 添加安装
# 设置安装在 ${CMAKE_INSTALL_PREFIX} 目录下
install(TARGETS dlib
    RUNTIME DESTINATION bin # 可执行文件
    LIBRARY DESTINATION lib # 动态库
    ARCHIVE DESTINATION lib # 静态库
    PUBLIC_HEADER DESTINATION include # 公共头文件，只有在设置了 PUBLIC_HEADER 时才会安装，或者使用下面一句
    )


#[[
运行，比如安装到 ~/Documents/course_lib 目录下，如果不设置 ，则会安装到 /usr/local 目录下
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/Documents/course_lib
cmake --build build
cmake --install build
]]