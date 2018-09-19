---
title: Qt+MySQL编程
date: 2018-08-12 18:19:00
categories: "Qt"
tags:
  - MySQL
  - C++
  - Qt
---
项目需要开发一个数据库软件，并且整个软件都是使用Qt搭建的，数据库选用的是MySQL，因此需要使用Qt调用MySQL，在此记录一下相关方法与问题，以方便自己与他人。  

本项目使用的是QtCreator(Qt5.5.0)+VisualStudio2013+MySQL5.7.17.0搭建。其他版本只要版本是正确对应的，都大同小异。
<!--more-->

# 准备工作
假设你已经正确安装了Qt和MySQL，并且已经将文件`../MySQL/MySQL Server 5.7/lib/libmysql.dll`复制到文件夹`../Qt5.5.0/5.5/msvc2013_64/bin`中，将文件`../MySQL/MySQL Server 5.7/lib/libmysql.lib`复制到文件夹`../Qt5.5.0/5.5/msvc2013_64/lib`中。  
之前遇到过MySQL安装过程中卡在MySQL Serve加载处，网络上的各种方法都没用，最后发现可能是使用了XX-Net等网络代理工具的问题，解决方法是**关闭网络防火墙**。

# Qt链接MySQL数据库
首先在xxx.pro工程文件中添加：
```
QT       += sql
```

在相应文件中引入相关头文件：
```c++
#include <QSql>
#include <QSqlQueryModel>
#include <QSqlDatabase>
#include <QSqlQuery>
```

在mainwindow.h文件的构造函数中添加：
```c++
QString hostName;
QString dbName;
QString userName;
QString password;
QSqlDatabase dbconn;
```

在mainwindow.cpp文件的构造函数中添加：
```c++
hostName = "localhost";   // 主机名
dbName = "wbq";   // 数据库名称
userName = "root";   // 用户名
password = "helloworld";   // 密码

dbconn = QSqlDatabase::addDatabase("QMYSQL");
dbconn.setHostName(hostName);
dbconn.setDatabaseName(dbName);
dbconn.setUserName(userName);
dbconn.setPassword(password);

qDebug("database open status: %d\n", dbconn.open());

QSqlError error = dbconn.lastError();
qDebug() << error.text();

dbconn.close();
```

如果数据库能够成功打开则调试窗口会出现以下信息：
```
database open status: 1

" "
```

主界面创建了数据库`dbconn`后，其他界面若需调用数据库只用定义一个子类成员函数：
```c++
void subwindow::setDbconn(QSqlDatabase *dbconn)
{
    this->dbconn = dbconn;
}
```

然后在mainwindow.cpp文件对应处调用该函数。

# Qt的MySQL数据库操作
## 读取MySQL数据库中某表格的字段名
通过以下代码可以将MySQL中某表格中的字段名读取到一个QSqlQueryModel中，读取model中的数据可以获取到字段名。
```c++
QSqlQueryModel *model_name = new QSqlQueryModel;
model_name->setQuery(QString("select COLUMN_NAME from information_schema.COLUMNS where table_name = 'your_table_name' and table_schema = 'your_db_name'"));
QModelIndex index_name = model_name->index(1,0);   // model为n行1列
qDebug()<<model_name->data(index_name).toString();
```

## 读取MySQL数据并输入到TableView中
MySQL数据库中数据可以看成一个model。假设要读取`wbq`数据库中的`cursheet`表，读取方式如下：
```c++
dbconn->open();
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery(QString("SELECT * FROM %1").arg("wbq.cursheet"));
ui->tableView->setModel(model);
dbconn->close();
```

## 读取MySQL多个表并根据主键值匹配融合成一个表
通常在读入数据后可能需要对多附表进行匹配操作，此时需要读取表中具体元素的数据，方法如下：
```c++
/*
定义一个用于存储匹配结果的model
*/
QStandardItemModel model_merge = new QStandardItemModel(ui->tableView)

/*
从wbq数据库中读取两个表进行匹配
*/
dbconn->open();
QSqlQueryModel *model_1 = new QSqlQueryModel;
model_1->setQuery(QString("SELECT * FROM wbq.sheet1"));
QSqlQueryModel *model_2 = new QSqlQueryModel;
model_2->setQuery(QString("SELECT * FROM wbq.sheet2"));

/*
将model_1中数据输入到model_merge中
*/
for(int i=0;i<model_1->rowCount();i++){
    for(int j=0;j<model_1->columnCount();j++){
        QModelIndex index = model_1->index(i,j);
        model_merge->setItem(i,j,new QStandardItem(model_1->data(index).toString()));
    }
}

/*
将model_2中的数据与model_merge进行匹配
(算法未优化，还望各位多包涵)
*/
QVector<int> nullRow_2; // 用于记录空值
for(int i=0;i<model_merge->rowCount();i++){
    QModelIndex index_m = model_merge->index(i,1);
    for(int j=0;j<model_2->rowCount();j++){
        QModelIndex index_2 = model_2->index(j,0);

        // 若两组数据键值相等，则将model_2数据扩展到model_merge已有列之后
        if(model_merge->data(index_m).toString()==model_2->data(index_2).toString()){
            for(int k=1;k<model_2->columnCount();k++){
                QModelIndex index_m2 = model_2->index(j,k);
                model_merge->setItem(i,k+22,new QStandardItem(model_2->data(index_m2).toString()));
            }
            break;
        }

        // 如果遍历完依然没有匹配上则存入nullRow_2
        if(j==(model_2->rowCount()-1)){
            nullRow_2.append(i);
        }
    }
}

// 删除nullRow_2中存放的无法匹配的数据
for(int i=nullRow_2.size();i>0;i--){
    model_merge->removeRow(nullRow_2[i-1]);
}
```

## 增加操作
假设要向`wbq`数据库中的`cursheet`表增加数据,`QVector<QString> validStringVect`为数据名称，`QVector<QLineEdit*> validLineEditVect`为数据值：
```c++
/*
创建语句头
*/
QString sqlStr = "";
sqlStr += QString("INSERT INTO wbq.%1 SET").arg(curSheet);

dbconn->open();

/*
为每一组数据在sqlstr中扩充相应的语句
*/
for (int i = 0; i < validStringVect.size(); ++i) {
    sqlStr += QString(" `%1`='%2'").arg(validStringVect.at(i)).arg(validLineEditVect.at(i)->text());
    if (i < validStringVect.size()-1) {
        sqlStr += ",";
    }
}

// 中文的话需要注意sqlstr的编码问题

/*
将语句sqlstr发送到MySQL，进行相应的操作
*/
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery(sqlStr);
model->setQuery(QString("SELECT * FROM %1").arg("wbq.cursheet")); // 刷新表格
dbconn->close();
```

## 删除操作
删除操作通过寻找主键完成对某一组数据的删除工作，具体实现如下所示：
```c++
/*
获取表格中的选中行索引（可为多行）
*/
std::set<int> rowSet;
QModelIndexList indexList = ui->tableView->getSelectedIndexs();
int indexNum = indexList.size();
for (int i = 0; i < indexNum; ++i) {
    rowSet.insert(indexList.at(i).row());
}

/*
根据主键值遍历删除选中行
*/
for (iter = rowSet.begin(); iter != rowSet.end(); ++iter) {
    QModelIndex indexToDel = model->index((*iter), 0);
    QString tmpSql;
    tmpSql = QString(QStringLiteral("DELETE FROM wbq.%1 WHERE index='%2'")).arg(curSheet).arg(indexToDel.data().toString());
    dbconn->open();
    QSqlQueryModel *model = new QSqlQueryModel;
    model->setQuery(tmpSql);
    dbconn->close();
}

/*
刷新表格
*/
dbconn->open();
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery(QString("SELECT * FROM %1").arg("wbq.cursheet"));
ui->tableView->setModel(model);
dbconn->close();
```

## 编辑操作
通过对TableView中某item数据修改完成对MySQL对应元素数据的修改，具体方法如下：
```c++
QString column_en[8] = {......} // 存储表格中的所有可编辑列列名

QModelIndex curIndex = ui->tableView->currentIndex(); // 获取当前选中的item的index
QAbstractItemModel *model = ui->tableView->model();
QModelIndex primaryIndex = model->index(curIndex.row(), 0); // 获取选中行第一列的index(主键值)
QString primaryKey = primaryIndex.data().toString(); // 获取选中行的主键值
QString editColumn = pColumn[curIndex.column()]; // 获取选中列列名

QString editStr = tmpLineEdit->text(); // item修改后的值

QString updateStr;
updateStr = QString(QStringLiteral("UPDATE wbq.%1 SET %2='%3' WHERE 主键值='%4'")).arg(this->curSheet).arg(editColumn).arg(editStr).arg(primaryKey);

dbconn->open();
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery(updateStr);
model->setQuery(QString("SELECT * FROM %1").arg("wbq.cursheet")); // 刷新表格
ui->tableView->setModel(model);
dbconn->close();
```

## 查询操作
软件中输入某列或某几列数据的上下限，完成查询功能。具体方法如下：
```c++
QString sqlStr = "";
sqlStr += QString("SELECT * FROM wbq.%1 WHERE ").arg(curSheet);

// 遍历各列数据的上下限
bool first = true;
for (int i = 0; i < searchField.size(); ++i) {
    if (!this->startLineEdit.at(i)->text().isEmpty()) {
        if (!first)
            sqlStr += QString(" AND ");
        sqlStr += QString(" `%1` > %2 ").arg(searchField.at(i)).arg(startLineEdit.at(i)->text());
        first = false;
    }
    if (!this->endLineEdit.at(i)->text().isEmpty()) {
        if (!first)
            sqlStr += QString(" AND ");
        sqlStr += QString(" `%1` < %2 ").arg(searchField.at(i)).arg(endLineEdit.at(i)->text());
            first = false;
    }
}
if (first) // 防止出现所有列上下限都没有设置的情况
    sqlStr += QString("1=1");
    dbconn->open();

QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery(sqlStr);
ui->tableView->setModel(model);
dbconn->close();
```

## 统计计算
项目要求软件实现对数据统计特征的计算功能，此处也是直接调用MySQL语句进行操作。具体方法如下：
```c++
dbconn->open();
QVector<QString> calStr = {"AVG", "STD", "MAX", "MIN"}; // 四种统计属性
QSqlQueryModel *queryModel = new QSqlQueryModel; // 获取各列统计信息的model
QVector<QVector<QString>> gVect; // 存储各列统计信息

QVector<QString> fieldStr; // 需要统计的列名(假设已赋值)

/*
获取queryModel
*/
for (int i = 0; i < calStr.size(); ++i) {
    QString queryStr = QString("SELECT ");
    for (int j = 0; j < fieldStr.size(); ++j) {
        queryStr.append(QString("%1(CAST(`%2` AS DECIMAL(8, 2)))").arg(calStr[i]).arg(fieldStr[j])); // 8代表数值长度，2代表小数位数
        if (j < fieldStr.size()-1)
            queryStr.append(QString(","));
    }
    queryStr.append(QString(" FROM wbq.%1").arg(this->curSheet));
    queryModel->setQuery(queryStr);

    /*
    queryModel读取到gVect中
    */
    int colCount = queryModel->columnCount();
    QVector<QString> rowVect;
    for (int j = 0; j < colCount; ++j) {
        rowVect.push_back(QString("%1").arg(queryModel->index(0, j).data().toFloat()));
    }
    gVect.push_back(rowVect);
}

dbconn->close();

/*
显示各列统计信息
*/
QAbstractTableModel *model = new QAbstractTableModel；
model->setDataVect(gVect);
QVector<QString> calStr_cn = {QStringLiteral("平均数"), QStringLiteral("方差"), QStringLiteral("最大值"), QStringLiteral("最小值")};
model->setHeaderVect(fieldStr, calStr_cn);
ui->tableView->setModel(model)； // 刷新表格
```

# Qt+MySQL发布
此步同样适用于一般的Qt软件发布。  
1). 单独将.exe执行程序放到一个空文件夹；  
2). 在`开始菜单`打开`Qt5.5 64-bit for Desktop (msvc 2013)`并`cd`命令到此文件夹；  
3). 运行命令`windeployqt 文件名.exe`，将生成的所有文件复制到`release`文件夹中；  
4). 将`libmysql.dll`复制到`release`文件夹中。

使用该软件的主机需要安装对应版本的MySQL，并将已存在的数据库导入。

# 注意事项
这里列写一下软件搭建过程中遇到的问题，以供参考。  
* 重装Qt后数据库无法连上：  
  将`release文件夹`中Qt的`.dll`用新装的Qt替代，哪怕Qt版本一致。

* 出现错误`cant connect mysql server on 127.0.0.1(10060)`，mysql workbench无法连接，但MySQL Serve显示正在运行：
  关闭网络防火墙。

* QT移植无法启动`This application failed to start because it could not find or load the Qt platform plugin`：  
  1). 将`../Qt5.5.0/5.5/msvc2013_64/bin`中的所有`.dll`复制到.exe目录下，尽管右下可能用不到；  
  2). 将文件夹`../Qt5.5.0/5.5/msvc2013_64/plugins/platforms`直接复制到.exe目录下。

* 在`Qt5.5 64-bit for Desktop (msvc 2013)`中输入`windeployqt 文件名.exe`时报错`Warning: Cannot find Visual Studio installation directory, VCINSTALLDIR is not set.`：  
  直接用 "VS2013 开发人员命令提示" 命令行去执行刚才的`windeployqt 文件名.exe`，会将 "vcredist_x64.exe"（vc x64 运行最少环境）程序放入当前目录。


---
# Reference:
[1] 在QT中使用MySQL数据库：https://blog.csdn.net/yunzhifeiti/article/details/72709140
[2] QT移植无法启动 This application failed to start because it could not find or load the Qt platform plugin：https://blog.csdn.net/jzwong/article/details/71479691
