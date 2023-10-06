# test_account/CMakeLists.txt

# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(test_account)

# 添加执行文件
add_executable(test_account test_account.cpp)

# 指定目标包含的头文件目录
target_include_directories(test_account PUBLIC "../account_dir")
# 添加库文件目录，如果不添加，找不到库文件
target_link_directories(test_account PUBLIC "../account_dir/build")
# 指定目标链接的库
target_link_libraries(test_account PRIVATE Account)

