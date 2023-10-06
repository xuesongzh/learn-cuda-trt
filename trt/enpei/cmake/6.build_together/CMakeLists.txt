#6.build_together/CMakeLists.txt`

# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(test_account)

# 添加动态库
add_library(Account SHARED "./account_dir/Account.cpp" "./account_dir/Account.h")

# 添加可执行文件
add_executable(test_account "./test_account/test_account.cpp")

# 添加头文件
target_include_directories(test_account PUBLIC "./account_dir")
# 添加链接库
target_link_libraries(test_account Account)