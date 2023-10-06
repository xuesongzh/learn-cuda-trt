#account_dir/CMakeLists.txt

# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(Account)

# 添加静态库，Linux下会生成libAccount.a
# Account是库名，STATIC表示静态库，SHARED表示动态库，后面是源文件
add_library(Account STATIC Account.cpp Account.h)