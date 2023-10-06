# C++环境配置及第一个程序

| 文档版本 | 更新人           | 更新时间   |
| -------- | ---------------- | ---------- |
| v1.0     | @恩培-计算机视觉 | 2022-12-07 |

[toc]

### 一、Visual Studio Code相关信息

* Visual Studio Code 下载地址：https://code.visualstudio.com/download
* VS Code建议安装插件列表：
  - 中文菜单：
    * MS-CEINTL.vscode-language-pack-zh-hans
  - SSH远程开发：
    * ms-vscode-remote.remote-ssh
    * ms-vscode-remote.remote-ssh-edit
    * ms-vscode.remote-explorer
  - C++开发
    * ms-vscode.cpptools
  - python开发
    * ms-python.python
  - 代码补全
    * TabNine.tabnine-vscode
    * GitHub.copilot
* VS Code SSH远程连接Ubuntu主机
  - 本地Ubuntu示例
  - autoDL示例：
    - autoDL地址：https://www.autodl.com/home
    - 省钱妙招：无卡启动（不挂载GPU，￥0.1/h左右）



### 二、Python开发环境配置

* 建议`conda`虚拟环境
* 测试代码`demo.py`：

```python
# python 代码测试

# 计算 1+2+3+4+5 的和
sum = 0;
for i in range(5):
    sum += i

# 打印结果
print(sum);
```
* debuger配置`.vscode`下`launch.json`添加

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            // "program": "${file}", // 当前文件
            "program": "demo.py", // 指定文件
            "console": "integratedTerminal",
            "justMyCode": true // false表示可以进入第三方库（如Pytorch）里进行调试
        }
    ]
}
```



### 三、C++ 开发环境配置

> 当前配置的环境主要为了演示C++基础知识教学，后面做项目时会有调整。

* 测试代码`main.cpp`：

```c++
#include <iostream>
using namespace std;

int main(){
    
    // 计算 1+2+3+4+5
    int sum {0};
    for (int i {0}; i < 5; i++){
        sum += i;
    }
    // 输出结果
    cout << sum << endl;
    return 0;
    
}
```

* 先用`g++  main.cpp -o main`生成可执行文件
* 再用VS Code 菜单：`终端-运行生成任务`生成可执行文件，需要在`.vscode`先添加`tasks.json`

> Linux中可以使用`which g++`确定`g++`的路径

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"type": "cppbuild",
			"label": "C/C++: g++ 生成活动文件",
			"command": "/usr/bin/g++", // g++的路径
			"args": [
				"-fdiagnostics-color=always", // 颜色
				"-g",  // 调试信息
				"-Wall", // 开启所有警告
				"-std=c++14", // c++14标准
				"${file}", // 文件本身，仅适用于C++基础知识教学，无法同时编译所有文件
				// "${fileDirname}/*.cpp", // 文件所在的文件夹路径下所有cpp文件
				"-o", // 输出
				"${workspaceFolder}/release/${fileBasenameNoExtension}" // 文件所在的文件夹路径/release/当前文件的文件名，不带后缀
			],
			"options": {
				"cwd": "${fileDirname}" // 文件所在的文件夹路径
			},
			"problemMatcher": [
				"$gcc"
			],
			"group": {
				"kind": "build",
				"isDefault": true
			},
			"detail": "编译器: /usr/bin/g++"
		}
	]
}
```

* 需要debuger，`launch.json`修改为：

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) 启动",
            "type": "cppdbg", // C++调试
            "request": "launch",
            "program": "${workspaceFolder}/release/${fileBasenameNoExtension}",  // 文件所在的文件夹路径/release/当前文件的文件名，不带后缀
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}", // 文件所在的文件夹路径
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为 gdb 启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description":  "将反汇编风格设置为 Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: g++ 生成活动文件" // tasks.json的label
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}", // 当前文件
            // "program": "demo.py", // 指定文件
            "console": "integratedTerminal",
            "justMyCode": true // false表示可以进入第三方库（如Pytorch）里进行调试
        }
    ]
}
```



### 四、第一个C++程序

```c++
#include <iostream>

int main(){
    int favorites_num;
    std::cout << "请输入0~10中你最喜欢的数字：" ;
    std::cin >> favorites_num;
    std::cout << favorites_num << "也是我喜欢的数字！" << std::endl;
  	return 0;
}
```



### 四、附录：vs code 中变量解释

```shell
以：/home/Coding/Test/.vscode/tasks.json 为例

${workspaceFolder} :表示当前workspace文件夹路径，也即/home/Coding/Test

${workspaceRootFolderName}:表示workspace的文件夹名，也即Test

${file}:文件自身的绝对路径，也即/home/Coding/Test/.vscode/tasks.json

${relativeFile}:文件在workspace中的路径，也即.vscode/tasks.json

${fileBasenameNoExtension}:当前文件的文件名，不带后缀，也即tasks

${fileBasename}:当前文件的文件名，tasks.json

${fileDirname}:文件所在的文件夹路径，也即/home/Coding/Test/.vscode

${fileExtname}:当前文件的后缀，也即.json

${lineNumber}:当前文件光标所在的行号

${env:PATH}:系统中的环境变量
```

