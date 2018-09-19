---
title: Qt+Python混合编程
date: 2018-08-13 14:04:00
categories: "Qt"
tags:
  - python
  - c++
  - Qt
---
项目使用Qt搭建了一个数据库软件，这个软件还需要有一些数据分析、特征重要度计算、性能预测等功能，而python的机器学习第三方库比较成熟，使用起来也比较便捷，因此这里需要用到Qt(c++)+python混合编程，在此记录一下相关方法与问题，以方便自己与他人。

本项目使用的是QtCreator(Qt5.5.0)+VisualStudio2013+python3.6.5搭建。其他版本只要版本是正确对应的，都大同小异。
<!--more-->

# 准备工作
假设你已经正确安装了Qt和python，由于Qt中的`slots`关键字与python重复，这里我们需要修改一下文件`../Anaconda/include/object.h`(注意先将原文件备份)：  
原文件(448行)：
```c
typedef struct{
    const char* name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    PyType_Slot *slots; /* terminated by slot==0. */
} PyType_Spec;
```
修改为：
```c
typedef struct{
    const char* name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    #undef slots // 这里取消slots宏定义
    PyType_Slot *slots;/* terminated by slot==0. */
    #define slots Q_SLOTS // 这里恢复slots宏定义与QT中QObjectDefs.h中一致
} PyType_Spec;
```

完成上述工作后我们需要在`.pro`文件中加入python的路径信息(我的python路径是`Y:/Anaconda`)：
```
INCLUDEPATH += -I  Y:/Anaconda/include

LIBS += -LY:/Anaconda/libs -lpython36
```

将`python3.dll`，`python36.dll`，`pythoncom36.dll`，`pywintypes36.dll`放到`.exe`目录下。

# Qt调用python脚本
## python文件
创建一个python脚本放在`release`项目目录下，这里我们新建了一个`kde.py`,其中包含无返回值函数`plotKDE(x, column, kernel, algorithm, breadth_first, bw, leaf_size, atol, rtol, title)`用于绘制核KDE曲线与直方图和有返回值函数`loadData()`用于读取本地`.csv`文件，图像绘制效果如下所示：
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/QtPython/Figure_1.jpg)

`kde.py`部分代码如下(为方便表达，后续步骤中我们将`plotKDE`函数简写为`plotKDE(x, column， kernel)`)：
```python
import csv
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

mpl.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, 'train.csv')

def plotKDE(x, column, kernel='gaussian', algorithm='auto', breadth_first=1,
            bw=30, leaf_size=40, atol=0, rtol=1E-8, title):
    # kde
    x_plot = np.linspace(min(x), max(x), 1000).reshape(-1, 1)
    x = np.mat(x).reshape(-1, 1)
    fig, ax = plt.subplots()
    kde = KernelDensity(bandwidth=bw, algorithm=algorithm, kernel=kernel,
                        atol=atol, rtol=rtol, breadth_first=breadth_first,
                        leaf_size=leaf_size).fit(x)
    log_dens = kde.score_samples(x_plot)
    ax.hist(x, density=True, color='lightblue')
    ax.plot(x_plot[:, 0], np.exp(log_dens))

    plt.title(title[column])
    plt.show()

def loadData():
    x = []
    with open(file_path, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            tmp = list(map(float, line[4:]))
            x.append(tmp)
    return x
```

## Qvector转pyObject类型
此处新建了一个类用于将Qt中存储的数据的`QVector<double>`类型等转换为用于python脚本的`QObject`类型。`QVector<QVector<double>>`的转换方法可以此类推。
```c++
PyObject *Utility::UtilityFunction::convertLabelData(QVector<double> *labels)
{
    int labelSize = labels->size();
    PyObject *pArgs = PyList_New(labelSize);

    for (int i = 0; i < labelSize; ++i) {
        PyList_SetItem(pArgs, i, Py_BuildValue("d", (*labels)[i]));
    }
    return pArgs;
}
```

## Qt调用python脚本
### 在Qt项目中调用`kde.py`的`plotKDE`函数显示图像(有输入，无返回值)：
```c++
// 初始化
Py_Initialize();
if (!Py_IsInitialized()) {
    printf("inititalize failed");
    qDebug() << "inititalize failed";
    return ;
}
else {
    qDebug() << "inititalize success";
}

// 加载模块，即loadtraindata.py
PyObject *pModule = PyImport_ImportModule("kde");
if (!pModule) {
    PyErr_Print();
    qDebug() << "not loaded!";
    return ;
}
else {
    qDebug() << "load module success";
}

// QVector<double> pKDE中存放了选中列的所有数据
PyObject *pKDEdata = Utility::UtilityFunction::convertLabelData(&pKDE); // 类型转换
PyObject *pArg = PyTuple_New(3);
PyTuple_SetItem(pArg, 0, pKDEdata);
// int column表示选中的列的索引
PyTuple_SetItem(pArg, 1, Py_BuildValue("i", column));
// Qstring kernel表示核类型
PyTuple_SetItem(pArg, 2, Py_BuildValue("s", kernel.toStdString().c_str()));

// 加载函数loadData()
PyObject *pFunc = PyObject_GetAttrString(pModule, "plotKDE");

if (!pFunc) {
    printf("get func failed!");
}
else {
    qDebug() << "get func success";
}

PyObject_CallObject(pFunc, pArg);

Py_DECREF(pModule);
Py_DECREF(pFunc);
Py_Finalize();
```

### 在Qt项目中调用`kde.py`的`loadData`函数读取本地数据(无输入，有返回值)：
```c++
Py_Initialize();
QVector<QVector<double> > *trainData; // 存储python脚本读入的数据

if (!Py_IsInitialized()) {
    printf("inititalize failed");
    qDebug() << "inititalize failed";
    return ;
}
else {
    qDebug() << "inititalize success";
}

// 添加当前路径(读文件的时候才需要)
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append('./')");

// 加载模块，即loadtraindata.py
PyObject *pModule = PyImport_ImportModule("kde");
if (!pModule) {
    PyErr_Print();
    qDebug() << "not loaded!";
    return ;
}
else {
    qDebug() << "load module success";
}

// 加载函数loadData()
PyObject *pLoadFunc = PyObject_GetAttrString(pModule, "loadData");

if (!pLoadFunc) {
    printf("get func failed!");
}
else {
    qDebug() << "get func success";
}

PyObject *retObjectX = PyObject_CallObject(pLoadFunc, NULL); // 获得python脚本返回数据

if (retObjectX == NULL) {
    qDebug() << "no return value";
    return ;
}

/*
将retObjectX导入trainData中(二维数据)
*/
int row = PyList_Size(retObjectX);
for (int i = 0; i < row; ++i) {
    PyObject *lineObject = PyList_GetItem(retObjectX, i);
    int col = PyList_Size(lineObject);
    QVector<double> tmpVect;
    for (int j = 0; j < col; ++j) {
        PyObject *singleItem = PyList_GetItem(lineObject, j);
        double item = PyFloat_AsDouble(singleItem);
        tmpVect.push_back(item);
    }
    trainData->push_back(tmpVect);
}
Py_Finalize();
```

# 注意事项
这里列写一下软件搭建过程中遇到的问题，以供参考。  
* 重装python的话别忘了修改`.pro`文件中的python路径；  
* importError:dll not load：  
  常见的matplotlib,numpy等DLL加载错误，通常是由python与对应的第三方包的版本不一致导致的。将anaconda文件下的`python.dll`和`python3.dll`文件拷贝到qt可执行文件exe同级目录下并覆盖。
* 多次调用`Py_Initialize()`和`Py_Finalize()`可能会出现异常：  
  最好在main.cpp里就输入`Py_Initialize()`，程序最后再`Py_Finalize()`。


---
# Reference:
[1] QT与Python混合编程经验记录：http://www.cnblogs.com/jiaping/p/6321859.html
[2] C++调用python浅析：https://blog.csdn.net/magictong/article/details/8947892
