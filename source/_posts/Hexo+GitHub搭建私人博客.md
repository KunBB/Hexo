---
title: Hexo+GitHub搭建私人博客
date: 2018-08-11 12:00:00
categories: "Blog"
tags:
  - blog
  - Hexo
  - GitHub
---
使用GitHub作为服务器搭建自己的私人博客相对来说成本更低且更容易实现，缺点是国内的搜索引擎无法检索到你的网页信息。GithUb提供了一个Github Pages的服务，可以为托管在Github上的项目提供静态页面。从零开始，博客具体搭建步骤如下：
<!--more-->

#  安装Git
这一步可以参考廖雪峰的Git教程：  
https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/00137396287703354d8c6c01c904c7d9ff056ae23da865a000  
一路默认安装即可。  
安装完成后，需要进一步进行设置，在命令行输入：
```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```
其中第一句的“名称”和第二句的“邮箱”替换成你自己的名字与E-mail地址，此步相当于注册你这台机器的信息。  
为了实现在Github中对Git仓库进行托管服务，我们需要先注册一个Github账号，然后创建SSH Key。打开Shell(Windows下打开Git Bash)，输入：
```
$ ssh-keygen -t rsa -C "youremail@example.com"
```
同样将邮箱地址换成你自己的邮箱地址。一路确认，会在用户主目录下创建一个.ssh的文件夹，里面有id_rsa和id_rsa.pub这两个文件，这两个就是SSH Key的秘钥对，id_rsa是私钥，不能泄露出去，id_rsa.pub是公钥，可以放心地告诉任何人。  
登录Github，打开“Account settings”，“SSH Keys”页面，然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴id_rsa.pub文件的内容：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/HexoGitHub/0.png?raw=true)
点“Add Key”，你就应该看到已经添加的Key：  
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/HexoGitHub/1.png?raw=true)

# 安装Node.js与Hexo
点击链接下载Node.js安装包：https://nodejs.org/dist/v8.11.3/node-v8.11.3-x64.msi  
一路默认安装即可，完成后打开控制台，如果安装成功会显示如下信息：  
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/HexoGitHub/2.jpg?raw=true)

安装Hexo。首先在本地创建一个用于写blog的文件夹，并使用`cd`命令进入该文件夹目录。在命令行输入`$ npm install hexo -g`安装Hexo。输入`hexo -v`检查Hexo是否安装成功：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/HexoGitHub/3.jpg?raw=true)
输入`hexo init`初始化该文件夹(漫长的等待...)，看到最后出现"Start blogging with Hexo!"即表示安装成功。接着输入`npm install`以安装所需的组件。

* 连接Hexo与Github(仅限于第一次搭建博客)  
在当前目录下找到并打开文件_config.yml，修改repository值(在末尾)：
```
deploy:
  type: git
  repository: git@github.com:KunBB/KunBB.github.io.git
  branch: master
```
repository值是你在Github项目里的ssh，如下图所示：
![Loading...](https://github.com/KunBB/MarkdownPhotos/blob/master/HexoGitHub/4.jpg?raw=true)
**注意blog中的所有配置文件名称与值之间都要有空格，否则不会生效。**  
如`type:git`错误，`type: git`正确。

# Hexo中使用Latex编写公式
## 安装Kramed
由于很多博客中会涉及到一些公式，而markdown本身的公式编写较为麻烦，作为一名科研工作者，Latex格式一定是相当熟悉的东西，因此我们需要通过安装第三方库来配置Hexo使用Latex格式书写公式。

Hexo默认的渲染引擎是marked，但是marked不支持mathjax，所以需要更换Hexo的markdown渲染引擎为hexo-renderer-kramed引擎，后者支持mathjax公式输出。
```
$ npm uninstall hexo-renderer-marked --save
$ npm install hexo-renderer-kramed --save
```

## 更改文件配置
打开文件"../node_modules/hexo-renderer-kramed/lib/renderer.js"进行修改(末尾)：
```
// Change inline math rule
function formatText(text) {
    // Fit kramed's rule: $$ + \1 + $$
    return text.replace(/`\$(.*?)\$`/g, '$$$$$1$$$$');
}

修改为：

// Change inline math rule
function formatText(text) {
    return text;
}
```

## 停止使用hexo-math并安装mathjax包
卸载hexo-math：
```
$ npm uninstall hexo-math --save
```
安装hexo-renderer-mathjax包:
```
$ npm install hexo-renderer-mathjax --save
```

## 更新Mathjax配置文件
打开文件"../node_modules/hexo-renderer-mathjax/mathjax.html"，将<script>的src修改为：  
"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"
即：<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

## 更改默认转义规则
因为LaTeX与markdown语法有语义冲突，所以Hexo默认的转义规则会将一些字符进行转义，所以我们需要对默认的规则进行修改。
打开文件"../node_modules/kramed/lib/rules/inline.js"：  
1.将
```
escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
```
更改为
```
escape: /^\\([`*\[\]()# +\-.!_>])/,
```
2.将
```
em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```
更改为
```
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

## 开启mathjax
打开文件"../Hexo/themes/next/\_config.yml"，找到含有"mathjax"的字段进行如下修改：
```
# MathJax Support
mathjax:
  enable: true
  per_page: true
  cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
```
写博客的时候需要在开头开启mathjax选项，添加以下内容：
```
title: LibSVM支持向量回归详解
date: 2018-01-30 10:10:00
categories: "SVM"
tags:
  -Machine learning
mathjax: true
```

# 上传博客
在Shell(Windows下打开Git Bash)中输入
```
$ hexo s
```
可在本地查看博客效果，默认端口号为4000，地址：http://localhost:4000/  
注意复制时不要用ctrl+c(会终止进程)。  
若页面一直无法跳转，可能是端口被占用，此时可以输入`hexo server -p 端口号`来改变端口号。  
在Shell(Windows下打开Git Bash)中输入
```
$ hexo d -g
```
可将本地博客推送到Github服务器，完成私人博客更新。

# 其他
## 博客图片管理
博客中所插入的图片需要图片链接，由于博客托管于Github服务器，所以如果博客中图片链接为国内网页的链接则可能存在图片加载缓慢甚至无法加载的现象。为解决这个问题我们可以在Github上新建一个Repo用于存放博客中所需要的图片，将图片的Github链接写入博客。

## 博客文件管理
当电脑重装系统或需要跟换电脑时，hexo+git往往需要重新配置，博客文件也需要复制到当前主机。因此比较方便的操作就是在Github上新建一个Repo用于存放blog目录的所有文件，当博客更新时，将最近版push到github。

---
# Reference:
[1] 使用Hexo+Github一步步搭建属于自己的博客（基础）：https://www.cnblogs.com/fengxiongZz/p/7707219.html  
[2] hexo+github搭建个人博客(超详细教程)：https://blog.csdn.net/ainuser/article/details/77609180  
[3] 如何在 hexo 中支持 Mathjax？：https://blog.csdn.net/u014630987/article/details/78670258  
[4] 使用LaTex添加公式到Hexo博客里：https://blog.csdn.net/Aoman_Hao/article/details/81381507
