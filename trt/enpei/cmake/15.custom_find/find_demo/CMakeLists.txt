# 最低版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(main)


# # 设置CMAKE_MODULE_PATH，以便find_package查找，cmake下需要有Finddlib.cmake文件
# set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/)
# message("cmake_module_path:${CMAKE_MODULE_PATH}")


# # 设置一个缓存变量，用于在命令行设置dlib的安装路径，给Finddlib.cmake使用
# set(DLIB_INSTALL_PATH "./" CACHE PATH "dlib的安装路径")
# message(STATUS "dlib的安装路径为：${DLIB_INSTALL_PATH}")


#使用find_package查找dlib(这里会从CMAKE_MODULE_PATH设置的路径中查找）
find_package(dlib REQUIRED)

if (dlib_FOUND)
    message("dlib found")
    message("dlib include dir: ${dlib_INCLUDE_DIR}")
    message("dlib lib: ${dlib_LIBRARY}")
    message("dlib version: ${dlib_VERSION}")
    message("dlib author: ${dlib_AUTHOR}")
    message("dlib lib dir: ${dlib_LIBRARY_DIR}")
else()
    message("dlib not found")
endif()


# 设置RPATH，否则install后，运行时找不到动态库
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
SET(CMAKE_INSTALL_RPATH "${dlib_LIBRARY_DIR}")

# 添加动态库
add_executable(main main.cpp)

# 添加头文件
target_include_directories(main PUBLIC ${dlib_INCLUDE_DIR})
# 链接动态库
target_link_libraries(main ${dlib_LIBRARY})


# 设置安装
install(TARGETS main 
	RUNTIME DESTINATION bin  # 可执行文件安装路径
)

#[[
CMAKE_INSTALL_PREFIX为安装路径（系统内置），DLIB_INSTALL_PATH为dlib安装路径

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./installed -DDLIB_INSTALL_PATH=~/Documents/course_lib
cmake --build build
cmake --install build

]]