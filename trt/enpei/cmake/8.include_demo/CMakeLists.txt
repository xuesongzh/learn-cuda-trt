cmake_minimum_required(VERSION 3.10)

project(include_demo)

message("调用include前的信息")

# include，引用一次就导入一次
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/module_1.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/module_1.cmake")

message("调用include后的信息")