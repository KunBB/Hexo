---
title: Qt+webservice的多线程实现
date: 2018-08-13 20:04:00
categories: "Qt"
tags:
  - Multithreading
  - Qt
  - webservice
  - xml
  - gsoap
---
项目使用Qt搭建了一个数据库软件，需要远程访问公司的MES系统，使用webservice技术进行通信并以XML格式传输数据，为了使网络监听过程中不影响主线程程序的正常运行，我们需要将webservice相关功能放在新开的独立线程中。

本项目使用的是QtCreator(Qt5.5.0)+VisualStudio2013+gSOAP2.8搭建。其他版本只要版本是正确对应的，都大同小异。
<!--more-->

# WebService相关概念
*参见参考文献[3]。*

WebService是一种跨编程语言和跨操作系统平台的远程调用技术。所谓跨编程语言和跨操作平台，就是说服务端和和客户端的搭建平台与编程语言可以都不相同。所谓远程调用，就是一台计算机a上 的一个程序可以调用到另外一台计算机b上的一个对象的方法。

其实可以从多个角度来理解WebService，从表面上看，WebService就是一个应用程序向外界暴露出一个能通过Web进行调用的API，也就是说能用编程的方法通过 Web来调用这个应用程序。我们把调用这个WebService的应用程序叫做客户端，而把提供这个WebService的应用程序叫做服务端。从深层次看，WebService是建立可互操作的分布式应用程序的新平台，是一个平台，是一套标准。它定义了应用程序如何在Web上实现互操作性，你可以用任何 你喜欢的语言，在任何你喜欢的平台上写Webservice ，只要我们可以通过Webservice标准对这些服务进行查询和访问。

WebService平台需要一套协议来实现分布式应用程序的创建。任何平台都有它的数据表示方法和类型系统。要实现互操作性，WebService平台必须提供一套标准的类型系统，用于沟通不同平台、编程语言和组件模型中的不同类型系统。Webservice平台必须提供一种标准来描述Webservice，让客户可以得到足够的信息来调用这个Webservice。最后，我们还必须有一种方法来对这个Webservice进行远程调用,这种方法实际是一种远程过程调用协议(RPC)。为了达到互操作性，这种RPC协议还必须与平台和编程语言无关。

XML+XSD,SOAP和WSDL就是构成WebService平台的三大技术。

* **XML+XSD**
WebService采用HTTP协议传输数据，采用XML格式封装数据(即XML中说明调用远程服务对象的哪个方法，传递的参数是什么，以及服务对象的 返回结果是什么)，XML是WebService平台中表示数据的格式，其优点在于它既是平台无关又是厂商无关的。
XML解决了数据表示的问题，但它没有定义一套标准的数据类型，更没有说怎么去扩展这套数据类型。例如整形数到底代表什么？16位，32位，64位？这些细节对实现互操作性很重要。XML Schema(XSD)就是专门解决这个问题的一套标准。它定义了一套标准的数据类型，并给出了一种语言来扩展这套数据类型。WebService平台就是用XSD来作为其数据类型系统的。当你用某种语言(如VB.NET或C#)来构造一个Webservice时，为了符合WebService标准，所有你使用的数据类型都必须被转换为XSD类型。你用的工具可能已经自动帮你完成了这个转换，但你很可能会根据你的需要修改一下转换过程。

* **SOAP**
SOAP是"简单对象访问协议"，SOAP协议 = HTTP协议 + XML数据格式。
WebService通过HTTP协议发送请求和接收结果时，发送的请求内容和结果内容都采用XML格式封装，并增加了一些特定的HTTP消息头，以说明HTTP消息的内容格式，这些特定的HTTP消息头和XML内容格式就是SOAP协议。SOAP提供了标准的RPC方法来调用WebService。  
SOAP协议定义了SOAP消息的格式，SOAP协议是基于HTTP协议的，SOAP也是基于XML和XSD的，XML是SOAP的数据编码方式。打个比喻：HTTP就是普通公路，XML就是中间的绿色隔离带和两边的防护栏，SOAP就是普通公路经过加隔离带和防护栏改造过的高速公路。

* **WSDL**
好比我们去商店买东西，首先要知道商店里有什么东西可买，然后再来购买，商家的做法就是张贴广告海报。 WebService也一样，WebService客户端要调用一个WebService服务，首先要有知道这个服务的地址在哪，以及这个服务里有什么方法可以调用，所以WebService务器端首先要通过一个WSDL文件来说明自己家里有啥服务可以对外调用，服务是什么（服务中有哪些方法，方法接受的参数是什么，返回值是什么），服务的网络地址用哪个url地址表示，服务通过什么方式来调用。  
WSDL(Web Services Description Language)就是这样一个基于XML的语言，用于描述WebService及其函数、参数和返回值。它是WebService客户端和服务器端都能理解的标准格式。因为是基于XML的，所以WSDL既是机器可阅读的，又是人可阅读的，这将是一个很大的好处。一些最新的开发工具既能根据你的Webservice生成WSDL文档，又能导入WSDL文档，生成调用相应WebService的代理类代码。  
WSDL文件保存在Web服务器上，通过一个url地址就可以访问到它。客户端要调用一个WebService服务之前，要知道该服务的WSDL文件的地址。 WebService服务提供商可以通过两种方式来暴露它的WSDL文件地址：1.注册到UDDI服务器，以便被人查找；2.直接告诉给客户端调用者。

# gSOAP总结
gsoap概念：是一种能够把C/C++语言的接口转换成基于soap协议的webservice服务的工具。从官网的说明文档可知gSOAP可以为我们完成以下工作：

1、自动生成C和C++源代码，以使用和部署XML、JSON REST API以及SOAP/XML API；
2、使用gSOAP的快速XML流处理模型进行XML解析和验证，实现实现可移植的快速的和精简的API，每秒处理10K+消息仅需几KB代码和数据；
3、将WSDL转换为有效的C或C++源代码以使用或部署XML Web服务；
4、将XML schemas(XSD)转换为高效的C或C++源代码，以使用gSOAP全面的XML schemas功能覆盖来使用或部署XML REST API；
5、为WSDL和/或XSD定义的大型复杂XML规范生成高效的C和C ++代码，例如eBay，ONVIF，HL7，FHIR，HIPAA 5010，CDISC，XMPP XEP，TR-069，AWS，EWS，ACORD，ISO 20022和SWIFT，FixML，XBRL，OTA，IATA NDC，FedEx等（您甚至可以将多个组合在一起）；
6、安全的XML处理过程：gSOAP不容易受到大部分XML攻击手段的攻击；
7、使用强大的XML数据绑定准确地序列化XML中的C和C++数据，这有助于通过静态类型快速开发类型安全的API，以避免运行时错误；
8、在WSDL和XSD文件上使用wsdl2h选项-O2或-O4进行“schema slicing”，通过自动删除未使用的schema类型（WSDL和XSD根目录中无法访问的类型）来优化XML代码库的大小；
9、使用和部署JSON REST API；
10、使用SAML令牌安全地使用HTTP/S，WS-Security和WS-Trust来使用和部署API;
11、使用测试信使CLI测试您的SOAP/XML API，它会自动生成基于WSDL的完整和随机化的SOAP/XML消息（使用带有选项-g的wsdl2h和soapcpp2）来测试Web服务API和客户端；
12、使用gSOAP Apache模块在Apache Web服务器中部署API，在IIS中使用gSOAP ISAPI扩展部署API；
13、使用gSOAP cURL插件和gSOAP WinInet插件来使用API；
14、符合XML的行业标准，包括XML REST和SOAP，WSDL，MIME，MTOM，DIME，WS-Trust，WS-Security（集成了WS-Policy和WS-SecurityPolicy），WS-ReliableMessaging，WS-Discovery，WS-Addressing等；

## gSOAP简介
gSOAP一种跨平台的开源的C/C++软件开发工具包。生成C/C++的RPC代码，XML数据绑定，对SOAP Web服务和其他应用形成高效的具体架构解析器，它们都受益于一个XML接口。 这个工具包提供了一个全面和透明的XML数据绑定解决方案，Autocoding节省大量开发时间来执行SOAP/XML Web服务中的C/C++。此外，使用XML数据绑定大大简化了XML自动映射。应用开发人员不再需要调整应用程序逻辑的具体库和XML为中心的数据。

## gSOAP结构
使用gSOAP首先需要用到了两个工具就是`../gsoap-2.8/gsoap/bin/win32/wsdl2h.exe`和`../gsoap-2.8/gsoap/bin/win32/soapcpp2.exe`，用于自动生成包含接口的c/c++源文件。

### wsdl2h.exe
该工具可以根据输入的wsdl或XSD或URL产生相应的C/C++形式的.h供`soapcpp2.exe`使用。示例如下：

新建一个文件夹，将`wsdl2h.exe`(和`.wsdl`文件)放入，命令行进入当前路径后输入以下命令：
```
wsdl2h [options] XSD and WSDL files ...

wsdl2h -o file.h file1.wsdl
wsdl2h -o file.h http://www.genivia.com/calc.wsdl
```
根据WSDL自动生成`file.h`头文件，以供`soapcpp2.exe`使用。

wsdl2h主要的运行选项如下：

|选项|描述|
|:-|:-|
|-a|对匿名类型产生基于顺序号的结构体名称|
|-b|提供单向响应消息的双向操作（双工）|
|-c|生成c代码|
|-c++|生成c++代码|
|-c++11|生成c++11代码|
|-d|使用DOM填充xs：any和xsd：anyType元素|
|-D|使用指针使具有默认值的属性成员可选|
|-e|不要限定枚举名称，此选项用于向后兼容gSOAP 2.4.1及更早版本，该选项不会生成符合WS-I Basic Profile 1.0a的代码。|
|-f|为schema扩展生成平面C++类层次结构|
|-g|生成全局顶级元素声明|
|-h|显示帮助信息|
|-I path|包含文件时指明路径，相当于#import|
|-i|不导入（高级选项）|
|-j|不生成SOAP_ENV__Header和SOAP_ENV__Detail定义|
|-k|不生成SOAP_ENV__Header mustUnderstand限定符|
|-l|在输出中包含许可证信息|
|-m|使用xsd.h模块导入基元类型|
|-N name|用name 来指定服务命名空间的前缀|
|-n name|用name 作为命名空间的前缀取代缺省的ns|
|-O1|通过省略重复的选择/序列成员来优化|
|-O2|优化-O1并省略未使用的模式类型（从根目录无法访问）|
|-o file|输出文件名|
|-P|不要创建从xsd__anyType继承的多态类型|
|-p|创建从base xsd__anyType继承的多态类型，当WSDL包含多态定义时，会自动执行此操作|
|-q name|使用name作为所有声明的C++命名空间|
|-R|在WSDL中为REST绑定生成REST操作|
|-r host[:port[:uid:pwd]]|通过代理主机，端口和代理凭据连接|
|-r:uid:pwd|连接身份验证凭据（验证身份验证需要SSL）|
|-s|不生成STL代码（没有std :: string和没有std :: vector）|
|-t file|使用类型映射文件而不是默认文件typemap.dat|
|-U|将Unicode XML名称映射到UTF8编码的Unicode C/C++标识符|
|-u|不产生工会unions|
|-V|显示当前版本并退出|
|-v|详细输出|
|-W|抑制编译器警告|
|-w|始终在响应结构中包装响应参数|
|-x|不生成_XML any/anyAttribute可扩展性元素|
|-y|为结构和枚举生成typedef同义词|

### soapcpp2.exe
该工具是一个根据.h文件生成若干支持webservice代码文件生成工具，生成的代码文件包括webservice客户端和服务端的实现框架，XML数据绑定等，具体说明如下：

新建一个文件夹，将`soapcpp2.exe`和`.h`文件放入，命令行进入当前路径后输入以下命令：
```
soapcpp2 [options] header_file.h

soapcpp2 mySoap.h
soapcpp2 -c -r -CL calc.h
```
会生成如下c类型文件：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/QtWebservice/Figure_1.jpg?raw=true)

使用命令`-i`会生成如下c++类型文件：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/QtWebservice/Figure_2.jpg?raw=true)

|文件|描述|
|:-|:-|
|soapStub.h|根据输入的.h文件生成的数据定义文件，一般我们不直接引用它|
|soapH.h|所有客户端服务端都应包含的主头文件|
|soapC.cpp|指定数据结构的序列化和反序列化方法|
|soapClient.cpp|用于远程操作的客户端存根例程|
|soapServer.cpp|服务框架例程|
|soapClientLib.cpp|客户端存根与本地静态(反)序列化器结合使用|
|soapServerLib.cpp|服务框架与本地静态(反)序列化器结合使用|
|soapXYZProxy.h|使用选项-i：c++服务端对象(与soapC.cpp和soapXYZProxy.cpp链接)|
|soapXYZProxy.cpp|使用选项-i：客户端代码|
|soapXYZService.h|使用选项-i：server对象(与soapC.cpp和soapXYZService.cpp链接)|
|soapXYZService.cpp|使用选项-i：服务端代码|
|.xsd|ns.XSD由XML Schema生成，ns为命名空间前缀名，我们可以看看是否满足我们的协议格式（如果有此要求）|
|.wsdl|ns.wsdl由WSDL描述生成|
|.xml|生成了几个SOAP/XML请求和响应文件。即满足webservice定义的例子message(实际的传输消息)，我们可以看看是否满足我们的协议格式(如果有此要求)|
|.nsmap|根据输入soapcpp2.exe的头文件中定义的命名空间前缀ns生成ns.nsmap，该文件包含可在客户端和服务端使用的命名空间映射表|

soapcpp的主要运行选项如下：

|选项|描述|
|:-|:-|
|-1|生成SOAP 1.1绑定|
|-2|生成SOAP 1.2绑定|
|-0|没有SOAP绑定，使用REST|
|-C|仅生成客户端代码|
|-S|仅生成服务端代码|
|-T|生成服务端自动测试代码|
|-Ec|为深度数据复制生成额外的例程|
|-Ed|为深度数据删除生成额外的例程|
|-Et|使用walker函数为数据遍历生成额外的例程|
|-L|不生成soapClientLib / soapServerLib|
|-a|使用SOAPAction和WS-Addressing来调用服务端操作|
|-A|要求SOAPAction调用服务端操作|
|-b|序列化字节数组char [N]为字符串|
|-c|生成纯C代码|
|-d <path>|保存到指定目录<path>中|
|-e|生成SOAP RPC编码样式绑定|
|-f N|多个soapC文件，每个文件有N个序列化程序定义（N≥10）|
|-h|打印一条简短的用法消息|
|-i|生成从soap struct继承的服务代理类和对象|
|-j|生成可以共享soap结构的C++服务代理类和对象|
|-I <path>|包含其他文件时使用，指明 < path > (多个的话，用`:'分割)，相当于#import ，该路径一般是gSOAP目录下的import目录，该目录下有一堆文件供soapcpp2生成代码时使用|
|-l|生成可链接模块（实验）|
|-m|为MEX编译器生成Matlab代码|
|-n|用于生成支持多个客户端和服务器端|
|-p <name>|生成的文件前缀采用< name > ，而不是缺省的 "soap"|
|-q <name>|使用name作为c++所有声明的命名空间|
|-r|生成soapReadme.md报告|
|-s|生成的代码在反序列化时，严格检查XML的有效性|
|-t|生成的代码在发送消息时，采用xsi:type方式|
|-u|通过抑制XML注释来取消注释WSDL / schema输出中的注释|
|-V|显示当前版本并退出|
|-v|详细输出|
|-w|不生成WSDL和schema文件|
|-x|不生成示例XML消息文件|
|-y|在示例XML消息中包含C / C ++类型访问信息|

# 实例介绍
## 功能介绍
1、客户端能够向服务端发送字符串数据；
2、服务端能够接收到客户端发送的字符串数据；
3、服务端对字符串数据解析、查重并录入数据库；
4、软件窗口显示连接状态。

最终效果如下所示：  
点击“开始连接”按钮：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/QtWebservice/Figure_4.png?raw=true)

从客户端接收到一组数据后：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/QtWebservice/Figure_5.png?raw=true)


## 实例步骤
**1. 生成源代码**
由于没有`.wsdl`文件，因此我们跳过`wsdl2h.exe`这一步骤，手动编写供`soapcpp2.exe`使用的头文件`mySoap.h`：
```c++
//gsoap ns service name: sendMsg
//gsoap ns service style: rpc
//gsoap ns service encoding: encoded
//gsoap ns service namespace: urn:sendMsg

int ns__sendMsg(char* szMsgXml,struct nullResponse{} *out);
```

新建文件夹，将`soapcpp2.exe`和`mySoap.h`放入，打开命令后进入该目录下，运行命令`soapcpp2 mySoap.h`，生成如下文件：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/QtWebservice/Figure_3.jpg?raw=true)

**2. 客户端程序编写**
在QtCreator中向客户端工程目录添加文件`sendMsg.nsmap`、`soapH.h`、`soapStub.h`、`stdsoap2.h`、`soapC.cpp`、`soapClient.cpp`、`stdsoap2.cpp`(`stdsoap2.h`和`stdsoap2.cpp`位于文件夹`../gsoap-2.8/gsoap`)。

客户端项目构建的是一个控制台程序，因此直接在main函数里进行编写代码：
```c++
#include <QCoreApplication>
#include "stdio.h"
#include "gservice.nsmap"
#include "soapStub.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    printf("The Client is runing...\n");

    struct soap *CalculateSoap = soap_new();

    char server_addr[] = "127.0.0.1:8080"; // url地址，IP号+设备端口号
    char* data = "flange,6,7,8,9,10,11,12,13"; // 所需传递的数据

    nullResponse result; // 用于sendMsg的空返回值
    int iRet = soap_call_ns__sendMsg(CalculateSoap,server_addr,NULL,data,&result); // 调用之前定义的ns__sendMsg方法(即服务端提供的方法)

    if ( iRet == SOAP_ERR){
        printf("Error while calling the soap_call_ns__sendmsg");
    }
    else
    {
        printf("Calling the soap_call_ns__add success。\n");
    }

    soap_end(CalculateSoap);
    soap_done(CalculateSoap);

    return a.exec();
}
```

**3. 服务端程序编写**
在QtCreator中向服务端工程目录添加文件`sendMsg.nsmap`、`soapH.h`、`soapStub.h`、`stdsoap2.h`、`soapC.cpp`、`soapServer.cpp`、`stdsoap2.cpp`。

由于服务端网络通信功能需要不断对端口进行监听，因此为避免影响软件其他功能的运行，在此需要新开一条线程。项目头文件源码如下：
socketconnect.h:
```c++
#ifndef SOCKETCONNECT_H
#define SOCKETCONNECT_H

#include <QWidget>
#include <QDebug>
#include <QVector>
#include "webservice_thread.h" // 多线程头文件
#include <QMessageBox>

namespace Ui {
class SocketConnect;
}

class SocketConnect : public QWidget
{
    Q_OBJECT

public:
    void setDbconn(QSqlDatabase *dbconn); // 主线程数据库
    explicit SocketConnect(QWidget *parent = 0);
    ~SocketConnect();

private slots:
    void pB_lj_clicked(); // 连接按钮
    void pB_dk_clicked(); // 断开连接按钮
    void linkState_accept(QVector<QString>); // 用于在窗口显示连接状态
    void data_accept(QVector<QString>); // 接收数据，检查后录入数据库

private:
    QSqlDatabase *dbconn;
    WebserviceThread *WebT;
    QVector<QString> v_linkstate;

    Ui::SocketConnect *ui;
};

#endif // SOCKETCONNECT_H
```

socketconnect.cpp:
```c++
#include "socketconnect.h"
#include "ui_socketconnect.h"
#include "scrollwidget.h"

SocketConnect::SocketConnect(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SocketConnect)
{
    ui->setupUi(this);

    ui->tabWidget->setTabText(1, QStringLiteral("法兰盘部件"));
    ui->tabWidget->setCurrentIndex(0);

    ui->lineEdit->setText("127.0.0.1"); // 暂时无用
    ui->lineEdit_2->setText("8080");
    ui->lineEdit_3->setText("helloworld");
    ui->lineEdit_4->setText("helloworld");
    ui->lineEdit_4->setEchoMode(QLineEdit::Password);

    ui->tableWidget_fla->setEditTriggers(QAbstractItemView::NoEditTriggers); // 表格不可编辑
    ui->tableWidget_fla->setSelectionBehavior(QAbstractItemView::SelectRows);

    ui->tableWidget_fla->setAlternatingRowColors(true);
    ui->tableWidget_fla->setStyleSheet("QTableView{alternate-background-color: rgb(183, 242, 238);}"); // 设置隔行换色

    connect(ui->pushButton_lj,SIGNAL(clicked(bool)),this,SLOT(pB_lj_clicked()));
    connect(ui->pushButton_dk,SIGNAL(clicked(bool)),this,SLOT(pB_dk_clicked()));
}

SocketConnect::~SocketConnect()
{
    delete ui;
}

void SocketConnect::setDbconn(QSqlDatabase *db){
    this->dbconn=db;
}

void SocketConnect::pB_lj_clicked()
{
    if(ui->lineEdit_3->text()=="helloworld"&&ui->lineEdit_4->text()=="helloworld"){
        WebT = new WebserviceThread(this);
        int port = ui->lineEdit_2->text().toInt();
        WebT->setParameters(port);
        connect(WebT,SIGNAL(linkState_send(QVector<QString>)),this,SLOT(linkState_accept(QVector<QString>)));
        connect(ui->pushButton_dk,SIGNAL(clicked(bool)),WebT,SLOT(stopclick_accept()));
        connect(WebT,SIGNAL(data_send(QVector<QString>)),this,SLOT(data_accept(QVector<QString>)));
        WebT->start(); // 进入线程
    }
    else{
        QMessageBox::critical(NULL, QString::fromLocal8Bit("警告"), QStringLiteral("账号或密码错误，无法连接！"), QMessageBox::Yes);
    }
}

void SocketConnect::linkState_accept(QVector<QString> link_state){
    ScrollWidget *s_linkstate = new ScrollWidget; // 自己定义的QWidget的子类，用于增加、更新、清空QList<QWidget*>

    QString linkstate;
    for(int i=0;i<link_state.size();i++){
        linkstate+=link_state[i];
    }
    v_linkstate.append(linkstate); // 存储线程文件传过来的连接状态

    for(int i=0;i<v_linkstate.size();i++){
        QLabel *label_linkstate = new QLabel(v_linkstate[i]);
        s_linkstate->addWidget(label_linkstate);
    }
    s_linkstate->updateWidget();
    ui->scrollArea->setWidget(s_linkstate); // 更新scrollarea
}

void SocketConnect::data_accept(QVector<QString> content){
    dbconn->open();

    QString dataClass = content[0]; // 第一个数据存储了数据类型信息
    content.erase(content.begin());

    //检查编号数据是否重复或为空
    dbconn->open();
    if(content[0].isEmpty()){
        QMessageBox::critical(NULL, QString::fromLocal8Bit("警告"), QStringLiteral("远程传输数据的编号为空，请检查并重新录入！"), QMessageBox::Yes);
        return;
    }
    QSqlQueryModel *model_sql = new QSqlQueryModel;
    model_sql->setQuery(QString("SELECT * FROM wbq.%1").arg(dataClass));

    for(int k=0 ;k<model_sql->rowCount(); k++){
        QModelIndex index = model_sql->index(k,0);
        if(content[0] == model_sql->data(index).toString()){
            QMessageBox::critical(NULL, QString::fromLocal8Bit("警告"), QStringLiteral("远程传输数据的编号已存在于数据库中，请检查并重新录入！"), QMessageBox::Yes);
            return;
        }
    }
    //结束检查

    if(dataClass=="flange"){
        ui->tabWidget->setCurrentIndex(1);

        //数据写入MySQL数据库
        QSqlQueryModel *model = new QSqlQueryModel;
        model->setQuery(QStringLiteral("INSERT INTO wbq.coordinator SET 编号=%1, ......").arg(content[0])......; // 输入数据
        dbconn->close();
        delete model;

        //数据显示到表格中
        int row = ui->tableWidget_fla->rowCount()+1;
        ui->tableWidget_fla->setRowCount(row);
        for(int i=0;i<content.size();i++){
            ui->tableWidget_fla->setItem(row-1,i,new QTableWidgetItem);
            ui->tableWidget_fla->item(row-1,i)->setText(content[i]);
        }
    }
    else{
        QMessageBox::critical(NULL, QString::fromLocal8Bit("警告"), QStringLiteral("信息格式有误或信息错误！"), QMessageBox::Yes);
    }
}

void SocketConnect::pB_dk_clicked()
{
    ui->scrollArea->takeWidget();

    QVector<QString> tmp;
    v_linkstate.swap(tmp);

    ui->tableWidget_fla->setRowCount(0);
}
```

多线程头文件webservice_thread.h:
```c++
#ifndef WEBSERVICE_THREAD
#define WEBSERVICE_THREAD

#include<QtSql/QSql>
#include<QtSql/qsqlquerymodel.h>
#include<QtSql/QSqlQuery>
#include<QtSql/qsqldatabase.h>
#include<QSqlError>
#include<QStandardItem>

#include <QThread>
#include <QDebug>
#include <QVector>
#include <QMetaType>
#include <QWaitCondition>

class WebserviceThread: public QThread{
    Q_OBJECT

public:
    WebserviceThread(QObject *parent=0);
    ~WebserviceThread();
    void run();
    void setParameters(int);

signals:
    void linkState_send(QVector<QString>); // 向主线程发送连接状态
    void data_send(QVector<QString>); // 向主线程发送从客户端接收的数据

private slots:
    void stopclick_accept(); // 改变runstate

private:
    QVector<QString> readXML(QString);
    int runstate; // 用于终止循环
    int nPort;
};

#endif // WEBSERVICE_THREAD
```

多线程文件webservice_thread.cpp：
```c++
#include "webservice_thread.h"
#include <sstream>
//gsoap文件
#include"gservice.nsmap"
#include"soapH.h"
#include"soapStub.h"
#include"stdsoap2.h"
#include"stdsoap2.cpp"
#include"soapC.cpp"
#include"soapServer.cpp"

QString Msg; // 存储客户端发送过来的数据

WebserviceThread::WebserviceThread(QObject *parent):QThread(parent){
    qRegisterMetaType<QVector<QString>>("QVector<QString>");
    runstate=0; //置1时停止网络循环
}

WebserviceThread::~WebserviceThread(){

}

void WebserviceThread::setParameters(int port){
    nPort=port;
}

int http_get_wbq(soap *);
void WebserviceThread::run(){
    struct soap wbq_soap;
    soap_init(&wbq_soap);
    wbq_soap.fget = http_get_wbq; // 网上有人说如果要传输的数据量大的话应该用http post
    int nMaster = (int)soap_bind(&wbq_soap,NULL,nPort,100); // 端口绑定
    if(nMaster<0)
        soap_print_fault(&wbq_soap,stderr);
    else{
        QVector<QString> link_state_1;
        link_state_1.append(QString("Socket connection successful: master socket = "));
        link_state_1.append(QString::number(nMaster, 10));
        emit linkState_send(link_state_1); // 发射初始连接后的状态信息

        for(int i=0;;i++){
            int nSlave = (int)soap_accept(&wbq_soap); // 端口监听，获取客户端连接信息
            if(nSlave<0){
                soap_print_fault(&wbq_soap,stderr);
                break;
            }

            QVector<QString> link_state_2;
            link_state_2.append(QString("Times "));
            link_state_2.append(QString::number(i, 10));
            link_state_2.append(QString(". Accepted connection from "));
            link_state_2.append(QString::number(int((wbq_soap.ip>>24)&0xFF), 10));
            link_state_2.append(QString("."));
            link_state_2.append(QString::number(int((wbq_soap.ip>>16)&0xFF), 10));
            link_state_2.append(QString("."));
            link_state_2.append(QString::number(int((wbq_soap.ip>>8)&0xFF), 10));
            link_state_2.append(QString("."));
            link_state_2.append(QString::number(int((wbq_soap.ip)&0xFF), 10));
            link_state_2.append(QString(": slave socket = "));
            link_state_2.append(QString::number(nSlave, 10));
            if(!runstate)
                emit linkState_send(link_state_2); // 发射客户端连接信息

            if(runstate)//点击断开连接后的下一次循环可以ping通，但无法调用服务端的sendMsg方法，即数据无法传送，不会造成数据丢失。
                break;

            if(soap_serve(&wbq_soap)!=SOAP_OK)
                soap_print_fault(&wbq_soap,stderr);
            soap_destroy(&wbq_soap);
            soap_end(&wbq_soap);

            QVector<QString> data = readXML(Msg); // 解析数据
            emit data_send(data); // 向主线程传递解析后的数据
            Msg="";
        }
    }
    soap_done(&wbq_soap);
}

void WebserviceThread::stopclick_accept(){
    runstate = 1;
}

/*
此功能是按自己格式解析的
*/
QVector<QString> WebserviceThread::readXML(QString Msg){
    QVector<QString> data;
    QStringList msglist = Msg.split(",");
    for(int i=0;i<msglist.size();i++){
        data.append(msglist[i]);
    }
    return data;
}

int ns__sendMsg(struct soap *soap, char* smsg, nullResponse* result)
{
    Msg = QString(smsg);
    return SOAP_OK;
}

/*
用于在网页页面正常显示信息（百度得到的解决方案）
*/
int http_get_wbq(struct soap *soap){
    soap_response(soap, SOAP_HTML); // HTTP response header with text/html
    soap_send(soap, "<HTML>My Web server is operational.</HTML>");
    soap_end_send(soap);
    return SOAP_OK;
}
```

## 存在问题
1、gSOAP中有对数据序列化与反序列化的功能，不应按自己的格式来传送数据；
2、断开连接按功能在问题，不能及时断开连接。点击“断开连接”按钮后，客户端需要再试图与服务端通信一次，服务端才能真正跳出循环，目前只能做到最后一次通信客户端报错，数据无法传送，不会造成数据丢失。

---
# Reference:
[1] Qt中多线程的使用：https://blog.csdn.net/mao19931004/article/details/53131872
[2] Qt使用多线程的一些心得——1.继承QThread的多线程使用方法：https://blog.csdn.net/czyt1988/article/details/64441443
[3] WebService学习总结(一)——WebService的相关概念：https://www.cnblogs.com/xdp-gacl/p/4048937.html
[4] gsoap使用总结：https://blog.csdn.net/byxdaz/article/details/51679117
