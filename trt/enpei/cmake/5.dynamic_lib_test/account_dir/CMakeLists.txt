#account_dir/CMakeLists.txt

# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(Account)

# 添加动态库，Linux下会生成libAccount.so
add_library(Account SHARED Account.cpp Account.h)