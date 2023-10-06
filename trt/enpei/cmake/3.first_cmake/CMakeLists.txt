# 单行注释
# 单行注释1

#[[
多行注释1
多行注释2
多行注释3
]]

# 指定版本
cmake_minimum_required(VERSION 3.10)

# 设置项目，项目名称、版本、描述、语言
project(main_test
        VERSION 1.0.0
        DESCRIPTION "main_test的项目描述"
        LANGUAGES CXX
)

# 设定target目标：可执行文件、库文件、自定义命令
add_executable(main_test main.cpp)

