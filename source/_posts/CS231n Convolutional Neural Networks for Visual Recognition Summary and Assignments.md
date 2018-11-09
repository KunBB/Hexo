---
title: CS231n Convolutional Neural Networks for Visual Recognition Summary and Assignments
date: 2018-09-24 11:10:00
categories: "CS231n"
tags:
  - Deep Leanring
  - Machine Learning
  - Artificial Intelligence
---
CS231n课程大家都很熟悉了，深度学习入门必备课程。这里就不多介绍了，只对课程资源进行归纳汇总，分享一下自己学习该课程后完成的作业，以供一起学习的同学们参考、交流。由于该课程的课件较为精炼，没有长篇大论，且知乎有全套的课件翻译，因此这里暂不对该课程知识点进行归纳总结，后续学习中如果有需要提炼的地方会对本文进行更新。
<!--more-->


# 课程资源
课程地址：http://cs231n.stanford.edu/
课程视频(EN, Spring 2017)：https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
课程地址(CN, Spring 2017)：http://www.mooc.ai/course/268
知乎课件翻译地址(Winter 2016)：https://zhuanlan.zhihu.com/p/22339097

# 课程作业
我在学习这门课程的时候，作业是从课程官网下载的，因此版本是Spring 2018。相较于之前的作业版本，Spring 2018 Assignment 2 中新加入了Layer Normalization和Group Normalization的内容以及PyTorch和Tensorflow的相关练习。　　
Assignments还在更新中，目前Assignments 2已经完成，后续会继续更新Assignments 3。课程Assignments详见：https://github.com/KunBB/cs231n_assignment

# 课程小结
## Lecture 1 ~ Lecture 7
见知乎的翻译课件，或是CS231n官网的英文课件；
## Lecture 8
主要讲解了一些主流的深度学习框架、PyTorch和Tensorflow的基本使用流程等，这部分内容通过完成最新的Assignment 2中的PyTorch或Tensorflow就差不多可以掌握了。
## [Lecture 9](http://xuyunkun.com/2018/09/28/CS231n%20Lecture%209%20CNN%20Architectures%20Summary/)
Lecture 9主要讲了一些经典的、比较流行的网络结构，详细讲解了AlexNet、ZFNet、VGGNet、GoogleNet和ResNet。
## [Lecture 10](http://xuyunkun.com/2018/10/01/CS231n%20Lecture%2010%20Recurrent%20Neural%20Networks/#more)
Lecture 10主要讲解了循环神经网络模型的结构、前向传播与反向传播，并由此引出了Attention机制、LSTM（长短期记忆网络）等，除此之外还介绍了图像标注、视觉问答等前沿问题。
## [Lecture 11](http://xuyunkun.com/2018/10/05/CS231n%20Lecture%2011%20Detection%20and%20Segmentation/#more)
Lecture 11主要讲解的是分割、定位与检测。具体包括语义分割、分类定位、目标检测和实例分割四部分。
## [Lecture 12](http://xuyunkun.com/2018/10/07/CS231n%20Lecture%2012%20Visualizing%20and%20Understanding/#more)
Lecture 12主要讲解的是对卷积神经网络的可视化和解释性的研究，从网络层的特征可视化开始到基于梯度提升方法的特征匹配、特征反演，进而衍生出纹理合成、图像生成和风格转移等。
## [Lecture 13](http://xuyunkun.com/2018/10/09/CS231n%20Lecture%2013%20Generative%20Models/#more)
Lecture 13主要讲解了无监督模型和生成模型，其中详细介绍了生成模型中的pixelRNN、pixelCNN、VAE、GAN等图像生成方法。
