# opencv  基本用法

| 作者             | 更新时间   | 版本  |
| ---------------- | ---------- | ----- |
| @恩培-计算机视觉 | 2023-02-03 | v 1.0 |



[toc]

## 一、vs code 结合Cmake debug

### 1.1 配置`tasks.json`

需要注意`"-DCMAKE_BUILD_TYPE=Debug" `要设置为`Debug`模式。

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			// cmake配置
			"type": "cppbuild",
			"label": "CMake配置",
			"command": "cmake", // cmake命令
			"args": [
				"-S .", // 源码目录
				"-B build", // 编译目录
				"-DCMAKE_BUILD_TYPE=Debug" // 编译类型
			],
			"options": {
				"cwd": "${workspaceFolder}" // 工作目录
			},
			"problemMatcher": [
				"$gcc"
			],
			"group": "build",
		},
		{
			// cmake编译
			"type": "cppbuild",
			"label": "CMake编译",
			"command": "cmake", // cmake命令
			"args": [
				"--build", // 编译
				"build", // 编译目录
			],
			"options": {
				"cwd": "${workspaceFolder}" // 工作目录
			},
			"problemMatcher": [
				"$gcc"
			],
			"group": "build",
			"dependsOn": [
				"CMake配置" // 依赖CMake配置，先执行CMake配置
			]
		},
		{
			// 删除build目录
			"type": "shell",
			"label": "删除build目录",
			"command": "rm -rf build",
			"options": {
				"cwd": "${workspaceFolder}" // 工作目录
			},
			"problemMatcher": [
				"$gcc"
			],
			"group": "build",
			
			
		}
	]
}
```

### 1.2 配置`launch.json`

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CMake调试",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/cmake_debug", // 编译后的程序，需要结合CMakeLists.txt中的add_executable()函数
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "CMake编译"
        }
    ]
}
```



## 二、图片、视频、摄像头读取显示

### 2.1 读取图片并显示

```c++
// 使用imread函数读取图片，和Python用法类似
// 读取的数据保存在Mat类型的变量image中，Mat是opencv中的图像数据结构，类似numpy中的ndarray
cv::Mat image = cv::imread("图片路径");

// 输出数据，以numpy和Python list格式输出
std::cout << cv::format(image, cv::Formatter::FMT_NUMPY) << std::endl;
std::cout << cv::format(image, cv::Formatter::FMT_PYTHON) << std::endl;

// 判断图像是否读取成功，返回true表示失败
if (image.empty())
{
  std::cout << "无法读取图片"  << std::endl;
  return 1;
} 
// imshow显示图像
cv::imshow("opencv demo", image);
// 保存图像
cv::imwrite("./output/gray_image.jpg", gray_image);

// 等待按键
cv::waitKey(0); 
```



### 2.2 读取视频文件并显示

```c++
// 读取视频：创建了一个VideoCapture对象，参数为视频路径
cv::VideoCapture capture("视频路径");

// 判断视频是否读取成功，返回true表示成功
if (!capture.isOpened())
{
  std::cout << "无法读取视频"  << std::endl;
  return 1;
}

// 读取视频帧，使用Mat类型的frame存储返回的帧
cv::Mat frame;
// 循环读取视频帧
while (true)
{
  // 读取视频帧，使用 >> 运算符或者read()函数，他的参数是返回的帧
  capture.read(frame);
  // capture >> frame;
  
  // 显示视频帧
  cv::imshow("opencv demo", frame);
}
```

### 2.3 读取摄像头并写入文件

```c++
// 读取视频：创建了一个VideoCapture对象，参数为摄像头编号
cv::VideoCapture capture(0);

// 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小  
cv::VideoWriter writer("record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 20, cv::Size(640, 480));

// 写入视频
writer.write(frame);
```



## 三、图片基本操作

### 3.1 颜色转换

```c++
// BGR -> Gray
// 三个参数分别是输入图像、输出图像、转换方式
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
// BGR -> HSV，Hue(色调)、Saturation(饱和度)、Value(明度)
cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
// BGR -> RGB
cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
```

### 3.2 图像filtering

```c++
// 三个参数分别是输入图像、输出图像、卷积核大小
cv::GaussianBlur(src, blur, cv::Size(7, 7), 0);
// 膨胀
// 三个参数分别是输入图像、输出图像、卷积核大小
cv::dilate(src, dilate, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
// 腐蚀
// 三个参数分别是输入图像、输出图像、卷积核大小
cv::erode(src, erode, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
```

### 3.3 形状调整

```c++
// ======== resize ========
// 三个参数分别是输入图像、输出图像、输出图像大小
cv::resize(src, resize, cv::Size(320, 240));

// ======== copy ========
cv::Mat copy;
src.copyTo(copy);
// ======== ROI裁剪 ========
cv::Rect rect(100, 100, 200, 100); // x, y, width, height
cv::Mat roi = src(rect);
cv::imwrite("./output/3.roi.jpg", roi);

// ======== 拼接 ========
cv::Mat dog_img = cv::imread("./media/dog.jpg");
cv::Mat dog_resize;
cv::resize(dog_img, dog_resize, cv::Size(320, 240));

// 水平拼接，需要保证两张图片的高度（rows）一致
cv::Mat hconcat_img;
cv::hconcat(resize, dog_resize, hconcat_img);
cv::imwrite("./output/3.hconcat.jpg", hconcat_img);

// 或者使用vector方式
std::vector<cv::Mat> imgs{resize, dog_resize, resize, dog_resize};
cv::Mat hconcat_img2;
cv::hconcat(imgs, hconcat_img2);
cv::imwrite("./output/3.hconcat2.jpg", hconcat_img2);

// 数组方式
cv::Mat imgs_arr[] = {dog_resize, resize, dog_resize, resize};
cv::Mat hconcat_img3;
cv::hconcat(imgs_arr, 4, hconcat_img3); // 4是数组长度
cv::imwrite("./output/3.hconcat3.jpg", hconcat_img3);

// 垂直拼接，需要保证两张图片的宽度（cols）一致
cv::Mat vconcat_img;
cv::vconcat(resize, dog_resize, vconcat_img);
cv::imwrite("./output/3.vconcat.jpg", vconcat_img);

// ======== 翻转 ========
cv::Mat flip;
// 三个参数分别是输入图像、输出图像、翻转方向
cv::flip(src, flip, 1); // 1表示水平翻转，0表示垂直翻转，-1表示水平垂直翻转

// ======== 旋转 ========
cv::Mat rotate;
// 三个参数分别是输入图像、输出图像、旋转角度
cv::rotate(src, rotate, cv::ROTATE_90_CLOCKWISE); // 顺时针旋转90度
```

### 3.4 绘制

```c++
// 创建一个黑色图像，参数分别是图像大小、图像类型，CV_8UC3表示8位无符号整数，3通道
cv::Mat image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);

// 绘制直线，参数分别是图像、起点、终点、颜色、线宽、线型
cv::line(image, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
// 绘制矩形，参数分别是图像、左上角、右下角、颜色、线宽、线型
cv::rectangle(image, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
// 绘制圆形，参数分别是图像、圆心、半径、颜色、线宽、线型
cv::circle(image, cv::Point(200, 150), 100, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
// 实心
cv::circle(image, cv::Point(200, 150), 50, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);

// ================== 多边形 ==================
cv::Point points[2][4]; // 定义两个多边形的顶点数组
// 第一个多边形的顶点
points[0][0] = cv::Point(100, 115);
points[0][1] = cv::Point(255, 135);
points[0][2] = cv::Point(140, 365);
points[0][3] = cv::Point(100, 300);
// 第二个多边形的顶点
points[1][0] = cv::Point(300, 315);
points[1][1] = cv::Point(555, 335);
points[1][2] = cv::Point(340, 565);
points[1][3] = cv::Point(300, 500);
// ppt[] 要同时添加两个多边形顶点数组的地址）
const cv::Point *pts_v[] = {points[0], points[1]};
// npts_v[]要定义每个多边形的定点数
int npts_v[] = {4, 4};
// 绘制多边形，参数分别是图像、顶点数组、顶点数、是否闭合、颜色、线宽、线型
cv::polylines(image, pts_v, npts_v, 2, true, cv::Scalar(255, 0, 255), 2, 8, 0);

// ================== 使用vector绘制多边形 ==================
std::vector<cv::Point> points_v;
// 随机生成5个点
for (int i = 0; i < 5; i++)
{
  points_v.push_back(cv::Point(rand() % 600, rand() % 600));
}
// 绘制多边形，参数分别是图像、顶点数组、是否闭合、颜色、线宽、线型
cv::polylines(image, points_v, true, cv::Scalar(255, 0, 0), 2, 8, 0);


// ================== 绘制文字 ==================
// 参数分别是图像、文字、文字位置、字体、字体大小、颜色、线宽、线型
cv::putText(image, "Hello World!", cv::Point(400, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2, 8, 0);
```

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img4.drawing.jpg?x-oss-process=style/wp" style="zoom:50%;" />

## 四、RTSP 视频流

### 4.1 本机构造RTSP视频流（optional）

```shell
# Ubuntu安装ffmpeg
sudo apt-get install ffmpeg

# 赋予权限
chmod +x rtsp-simple-server
chmod +x start_server.sh
# 运行服务
./start_server.sh

# 退出服务
pkill rtsp-simple-server
pkill ffmpeg
```

### 4.2 使用ffmpeg作为视频解码

```c++
// CAP_FFMPEG：opencv 使用ffmpeg解码
cv::VideoCapture stream1 = cv::VideoCapture("rtsp地址", cv::CAP_FFMPEG);
```



## 五、人脸检测小例子

附件位置：`5.face_detection`

