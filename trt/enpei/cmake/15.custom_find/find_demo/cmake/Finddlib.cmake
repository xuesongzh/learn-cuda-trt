#寻找 dlib.h
find_path(dlib_INCLUDE_DIR dlib.h PATHS ${DLIB_INSTALL_PATH}/include)

#寻找 libdlib.so
find_library(dlib_LIBRARY dlib  PATHS ${DLIB_INSTALL_PATH}/lib)

# 如果dlib_INCLUDE_DIR和dlib_LIBRARY都找到了，那么就设置dlib_FOUND为TRUE
if(dlib_INCLUDE_DIR AND dlib_LIBRARY)
    set(dlib_FOUND TRUE) 
    set(dlib_VERSION 1.0.0) # dlib的版本号
    set(dlib_AUTHOR "enpei") # dlib的作者
    # lib文件所在目录
    get_filename_component(dlib_LIBRARY_DIR ${dlib_LIBRARY} DIRECTORY)
endif()