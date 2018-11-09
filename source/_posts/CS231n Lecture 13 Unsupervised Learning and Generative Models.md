---
title: CS231n Lecture 13 Unsupervised Learning and Generative Models
date: 2018-10-09 13:27:00
categories: "CS231n"
tags:
  - Deep Leanring
  - Machine Learning
  - Artificial Intelligence
mathjax: true
---
Lecture 13主要讲解了无监督模型和生成模型，其中详细介绍了生成模型中的pixelRNN、pixelCNN、VAE、GAN等图像生成方法。
<!--more-->

# Unsupervised Learning
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/1.png)
监督式学习我们都很熟悉了，我们有数据x和标签y，我们在的目的是学习到一个函数可以将数据x映射到标签y，标签可以有很多形式。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/2.png)
无监督学习要做的就是在我们拥有的只是一些没有标签的训练数据的情况下学习一些数据中潜在的隐含结构。

典型的无监督学习有一下几种：
1. 聚类
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/3.png)
聚类的目标是找到数据中的分组，组内的数据在某种度量方式的比较下是相似的。

2. 降维
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/4.png)
降维的目标是找出一些轴，在这些轴向上训练数据的方差最大。这些轴向就是数据潜在结构的一部分。我们可以用这些轴来减少数据维度，数据在每个保留下来的维度上都有很大的方差。

3. 学习数据的特征表达
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/5.png)
在监督式方法中，如分类任务中我们使用了监督式的损失函数，我们有数据标签，我们可以训练一个神经网络，我们可以把激活函数理解为数据的某种未来表征。而在无监督学习中，如自动编码器，损失函数的目标是重构输入数据，然后通过这个重构来学习表征。

4. 密度估计
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/6.png)
此项任务中我们想要估计数据的内在分布情况。如上图1中我们有一些一维的点，我们尝试用高斯函数来拟合这一密度分布情况。如上图2，是一个二维数据，我们尝试估计密度分布，并且可以为该密度函数建模。

# Generative Models
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/7.png)
生成模型的任务是给定训练数据，从相同的数据分布中生成新的样本。我们有一些训练数据，这些训练数据是由某种分布p-data中生成的，我们想从中学习到一个模型p-model来以相同的分布生成样本，即我们希望p-model能学的和p-data尽可能地相似。

生成模型可以解决密度估计问题。我们可以使用生成式模型来做显式的密度估计，此种情况我们会求解出目标模型p-model。或者我们也可以进行隐式的密度估计，这种情况下我们会习得一个能够从p-model中生成样本的模型而不需要显式地定义它。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/8.png)
生成模型可以从数据分布中创造出我们想要的真实样本。上图左边为生成的图片，中间为生成的人脸，除此之外还可以做超分辨率或者着色之类的任务。另外，我们还可以用关于时间序列数据的生成模型来进行仿真和规划，这样一来就能在强化学习应用中派上用场。同时训练生成式模型也能使得隐式表征的推断成为可能。

**生成模型分类**：
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/9.png)

## PixelRNN and PixelCNN
这些都属于全可见信念网络。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/10.png)
目的是对一个密度分布显示建模。我们有图像数据$x$，我们希望对该图像的概率分布或者似然$p(x)$建模。于是我们使用链式法则将这一似然分解为一维分布的乘积。我们有每个像素$x_i$的条件概率，其条件是给定所有下标小于$i$的像素（$x_1$ ~ $x_{i-1}$）。此时图像中所有像素的概率或者联合概率就是所有这些像素点，这些似然的乘积。一旦我们定义好这一似然，为了训练这一模型，我们只要在该定义下最大化我们的训练数据的似然。

那么如果我们观察下右边的像素值概率分布，可以发现这是一个十分复杂的分布。我们之前已经了解到如果想要进行一些复杂的变换，我们可以利用神经网络来实现这一映射。

### PixelRNN
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/11.png)
该方法从左上角像素开始，一个接一个地生成像素，顺序如图所示。序列中每一个对之前像素的依赖关系都会通过LSTM来建模。

缺点：这种方法是顺序生成的，速度会很慢。


### PixelCNN
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/12.png)
我们依然是从角落开始并展开，进而生成整张图片。区别在于，现在使用CNN替代RNN来对所有这些依然关系建模。

 现在我们打算在环境区域（图示中间这个特定像素点的附近区域）上使用CNN，生成该区域包围的这个像素。取待生成像素点周围的像素（已生成像素区域内的灰色区域），把他们传递给CNN用来生成下一个像素值。每一个像素位置都有一个神经网络输出，该输出将会是像素的softmax损失值。我们通过最大化训练样本图像的似然来训练模型，在训练的时候取一张训练图像来执行生成过程，每个像素位置都有正确的标注值，即训练图片在该位置的像素值，该值也是我们希望模型输出的值。

Q：这里用到了标签值，为何还是无监督学习？
A：我们并没有为了训练图像而收集数据，我们只是将输入数据作用于模型末端函数。


![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/13.png)
PixelCNN训练要比pixelRNN更快，因为在每一个像素位置，我们想要最大化我们已有的训练数据的似然，我们已经有了所有的值，这些值来自训练数据，所以我们可以训练的更快，但在测试时的生成环节，我们想要从第一个像素生成一个全新的图像，我们并不需要在生成时做任何学习，只需逐一生成像素点，因此生成时间仍然很慢。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/14.png)
PixelRNN和pixelCNN能显示地计算似然$p(x)$，这是一种可以优化的显式密度模型。该方法同时给出了一个很好的评估度量，你可以通过你所能计算的数据的似然来度量出你的样本有多好，同时这些方法能够生成相当不错的样本。

这些方法的主要缺陷在于，由于生成过程是序列化的，因此速度上会很慢。

## Variational Autoencoders (VAE)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/15.png)
PixelCNN定义了一个易于处理的密度函数，我们可以直接优化训练数据的似然。对于变分自编码器我们将定义了一个不易处理的密度函数，现在我们通过附加的隐变量$z$对密度函数进行建模。我们数据的似然$p(x)$是等式右边的积分形式，即对所有可能的$z$值取期望，我们无法直接优化它，我们只能找出一个似然函数的下界然后再对该下界进行优化。

### Autoencoders
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/16.png)
变分自编码器与自动编码器的无监督模型息息相关。我们不通过自动编码器来生成数据，它是一种利用无标签数据来学习低维特征表示的无监督学习。我们有输入数据$x$，我们想要学习一些特征$z$，我们会有一个编码器进行映射，来实现从该输入数据到特征$z$的映射。该编码器可以有多种不同的形式，常用的是神经网络。最先提出的是非线性层的线性组合，又有了更深的全连接网络，又出现了CNN。我们取得输入数据$x$然后将其映射到某些特征$z$，我们通常将$z$限制在比$x$更小的维度上，由此可以实现降维。降维的目的是希望捕捉到的是重要的特征。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/17.png)
自动编码器将该模型训练成一个能够用来重构原始数据的模型。我们用编码器将输入数据映射到低维的特征$z$（也就是编码器网络的输出），同时我们想获得这些基于输入数据得到的特征，然后用第二个网络也就是解码器网络输出一些跟$x$有相同维度并和$x$相似的东西，也就是重构原始数据。对于解码器，我们一般使用和编码器相同类型的网络（通常与编码器对称）。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/18.png)
流程是我们取得输入数据，把它传给编码器网络（比如一个四层的卷积网络），获取输入数据的特征，把特征传给解码器（四层的解卷积网络），在解码器末端获得重构的数据。

至于为何选用卷积网络作为编码器而用解卷积网络作为解码器。事实上对于解码器来说，这是因为在编码器那里高维的输入被映射到低维的特征，而现在我们需要反其道而行。也就是从低维特征回到高维的重构输入（这里可以参考一下[Lecture 11](https://blog.csdn.net/qq_29176963/article/details/82928426)中关于反卷积的讲解）。

为了能够重构输入数据的效果，我们使用类似L2损失函数，让输入数据中的像素与重构数据中的像素相同。这里虽然有损失函数，但我们没有使用任何外部标签来训练模型。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/19.png)
一旦我们训练好模型，我们可以去掉**解码器**。这么做是为了生成重构的输入信息，同时为了计算损失函数。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/20.png)
使用训练好的**编码器**实现特征映射，而我们可以利用此特性来初始化一个监督式模型。通过编码器得到输入数据的特征，顶部有一个分类器，如果是分类问题我们可以用它来输出一个类标签，在这里使用了外部标签和标准的损失函数如softmax。

这么做的价值在于我们可以用很多无标签数据来学习到很多普适特征表征。可以用学习到的特征表征来初始化一个监督学习问题，因为在监督学习的时候可能只有很少的有标签训练数据，而少量的数据很难训练模型，可能会出现过拟合等其他一些问题，通过使用上面得到的特征可以很好地初始化网络。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/21.png)
我们已经见识了自动编码器重构数据，学习数据特征，初始化一个监督模型的能力。这些学习到的特征同时也具有能捕捉训练数据中蕴含的变化因素的能力。我们获得了一个含有训练数据中变化因子的隐变量$z$。

那么我们能用自动编码器生成新的图像吗？

### Variational Autoencoders
这是通过向自编码器中加入随机因子获得的一种模型，这样我们就能从该模型中采样从而生成新的数据。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/22.png)
我们有训练数据$x_i$（$i$的范围从1到N），数据$x$是从某种潜在的不可观测的隐式表征$z$中生成的。$z$的元素要捕捉的信息是训练数据中某种变化因子的多少，是某种类似于属性的东西。例如我们想要生成微笑的人脸，$z$代表的就是脸上有几分笑意、眉毛的位置、嘴角上扬的弧度等。

**生成过程**：从$z$的先验分布中采样，对于每种属性，我们都假设一个我们觉得它应该是一个怎样的先验分布。高斯分布就是一个对$z$中每个元素的一种自然的先验假设。同时我们会通过从在给定z的条件下，$x$的条件概率分布$p(x|z)$中采样。先对$z$采样，也就是对每个隐变量采样，接下来利用它对图像$x$采样。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/23.png)
对于上述采样过程，真实的参数是$\theta^*$，我们有关于先验假设和条件概率分布的参数，我们的目的在于获得一个生成式模型，从而利用他来生成新的数据，真实参数中的这些参数是我们想要估计并得出的。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/24.png)
如何表述上述模型：对上述过程建模，选一个简单的关于$z$的先验分布，例如高斯分布。对于给定$z$的$x$的条件概率分布$p(x|z)$很复杂，所以我们选择用神经网络来对$p(x|z)$进行建模。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/25.png)
我们想要训练好该模型，这样一来就能习得一个对于这些参数的估计。训练网络时，我们需要调用解码器网络，选取隐式特征并将其解码为它所表示的图像。一个直接且自然的训练策略是通过最大化训练数据的似然函数来寻找这些模型的参数。在已经给定隐变量z的情况下，我们需要写出$x$的分布$p$并对所有可能的$z$值取期望，因为z值是连续的所以表达式是一个积分。

现在我们想要最大化它的似然，但发现这一积分很难求解。

#### Variational Autoencoders: Intractability
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/26.png)
似然项的第一项是$z$的分布$p(z)$，这里如前所述，它可以被直接设定为高斯分布。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/27.png)
对于$p(x|z)$我们之前说要指定一个神经网络解码器，任意给定一个$z$，我们就能获得$p(x|z)$。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/28.png)
如果我们想要对每一个z值计算$p(x|z)$是很困难的，我们无法计算该积分。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/29.png)
似然难解直接导致了模型的其他项，如后验概率$p(z|x)$也是难解的。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/59.png)
因此我们无法直接进行优化。一个可以让我们训练该模型的办法是，如果在使用解码器网络来定义一个对$p(x|z)$建模的神经网络的同时，额外定义一个编码器$q(z|x)$，将输入$x$编码为$z$，从而得到似然$p(z|x)$。也就是说我们定义该网络来估计出$p(z|x)$，这个后验密度分布项仍然是难解的，我们用该附加网络来估计该后验分布，这将使我们得到一个数据似然的下界，该下界易解也能优化。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/30.png)
与自编码器类似，在变分自编码器中我们想得到一个生成数据的概率模型，将输入数据$x$送入编码器得到一些特征$z$，然后通过解码器网络把z映射到图像$x$。

我们这里也有编码器网络和解码器网络，但是我们要将一切参数随机化。参数是$\phi$的编码器网络$q(z|x)$输出一个均值和一个对角协方差矩阵；解码器网络$p(x|z)$输入$z$，输出均值和关于$x$的对角协方差矩阵。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/31.png)
为了得到给定$x$下的$z$和给定$z$下的$x$，我们会从这些分布（$p$和$q$）中采样。现在我们的编码器和解码器网络所给出的分别是$z$和$x$的条件概率分布，并从这些分布中采样从而获得值。

编码器网路也是一种识别或推断网络，因为是给定$x$下对隐式表征$z$的推断；解码器网络执行生成过程，所以也叫生成网络。

#### Variational Autoencoders: Solution process
既然已经有了编码器和解码器网络，现在我们求解数据似然，这里我们会使用对数似然。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/32.png)
如果我们想获得$\log(p(x))$，我们要对其关于$z$取期望，z是采样自分布$q(z|x)$，也就是我们通过编码器网络定义的分布。我们之所以这么做是因为$p(x)$并不依赖于$z$。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/33.png)
根据贝叶斯公式，从这一原始的表达式开始，我们可以将其展开为上式。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/34.png)
下面为了求解它，我们可以再乘一个常数。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/35.png)
这样一来，我们要做的就是把上式写成三项之和的形式，这三项都是有意义的。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/36.png)
仔细观察，会发现第一项是$\log(p(x|z))$关于$z$取期望，接下来有两个KL散度项，用来描述这两个分布有多么相似，也就是$q(z|x)$和$p(z)$有多相似，是对分布函数距离的度量。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/37.png)
再仔细观察一下。第一项是$p(x|z)$，是由解码器网络提供的。同时我们能够通过采样计算并估计出这些项的值，而且我们还会看到我们能够通过一种叫做重参数化的技巧来进行一次可微分的采样。

第二个KL项是两个高斯分布之间的KL散度。$q(z|x)$是由我们的编码器网络生成的一个性质很好的高斯分布（由编码器生成的均值和协方差构成）。同样，我们的先验假设$p(z)$也是一个高斯分布。我们有一个关于两个高斯分布的KL散度就等于我们获得了一个很好的闭式解。

第三个KL散度是关于$q(z|x)$和$p(z|x)$。我们知道$p(z|x)$是一个难解的后验概率，这一项依然是个问题。但是KL散度是对两个分布之间距离的度量，从定义上看它总是大于等于0。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/38.png)
因此我们能做的是在这里前两项可以很好地求解，而第三项一定大于等于0。因此，前两项合起来就是一个可以求解的下界，我们就可以对其取梯度并进行优化。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/39.png)
那么为了训练一个变分自编码器，我们需要优化并最大化这一下界，因此我们是在优化对数似然的下界。

另一个对于下界的解释是，第一项是对所有采样的$z$取期望，$z$是$x$经过编码器网络采样得到的，对$z$采样然后再求所有$z$对应的$p(x|z)$。让$p(x|z)$变大，就是最大限度地重构数据。第二项是让KL的散度变小，让我们的近似后验分布和先验分布变得相似，意味着我们想让隐变量$z$遵循我们期望的分布类型和形状。

Q：为何将先验分布即隐变量分布设定为高斯分布？
A：我们是在定义某种生成过程，该过程首先要对$z$采样然后对$x$采样。把它假设为高斯分布是因为这是一种合理的先验模型，对于隐变量的属性来说，分布成某种高斯分布式讲得通的，而且这么做可以让我们接下来能够优化模型。

#### Variational Autoencoders: Training process
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/40.png)
左上角的公式就是我们要优化及最大化的下界。现在对于前向传播而言我们要按下面的流程处理。
1.	我们有输入数据$x$；
2.	让输入数据传递经过编码器网络，得到$q(z|x)$；
3.	通过$q(z|x)$来计算公式中的KL项；
4.	根据给定$x$的$z$分布对$z$进行采样，由此获得了隐变量的样本；
5.	把$z$继续传给解码器网络，得到$x$在给定$z$条件下的分布的两个参数（均值和方差）；
6.	最终可以在给定$z$的条件下从这个分布中采样获得$x$，并会产生一些样本输出。

在训练的时候，我们就是要获得该分布，而我们的损失项将会是给定$z$条件下对训练像素值取对数。那么我们的损失函数要做的就是最大化被重构的原始输入数据的似然。

现在对于每个小批量输入，我们都要计算这一前向传播过程，取得所有我们所需的项，它们都是可微分的，所以我们接下来把它们全部反向传播回去获得梯度。我们利用梯度不断更新参数，包括编码器和解码器、网络参数$\theta$和$\phi$，从而最大化训练数据的似然。

#### Variational Autoencoders: Generation process
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/41.png)
一旦我们训练好VAE，要想生成数据，我们只需要解码器网络。我们在训练阶段就开始对$z$采样，而不用从后验分布中采样。在生成阶段，会从真实的生成过程中采样。先从设定好的先验分布中采样，接下来从这里对数据$x$采样。

在本例中通过在MNIST数据集上训练VAE，我们可以生成这些手写数字样本。我们用$z$表示隐变量，因为是从先验分布的不同部分采样，所以我们可以通过改变$z$来获得不同的可解释的意义。这里可以看到一个关于二维z的数据流形。如果我们有一个二维的$z$然后我们让$z$在某个区间内变化，比如该分布的百分比区间，接下来让$z_1$和$z_2$逐渐变化，从这幅图中可以看到各种不同的$z_1$和$z_2$的组合所生成的图像，它会在所有这些不同的数字之间光滑地过渡变化。


![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/42.png)

我们对$z$的先验假设是对角的，这样做是为了促使它成为独立的隐变量，这样它才能编码具有可解释性的变量。因此我们就有了$z$的不同维度，他们编码了不同的具有可解释性的变量。

在人脸数据上训练的模型中，随着我们改变$z_1$，从上往下看笑脸的程度在逐渐改变，从最上面的眉头紧锁到下面大的笑脸；接下来改变$z_2$，从左往右看发现人脸的朝向在变化，从一个方向一直向另一个方向变化。

$z$同时也是很好的特征表示，因为$z$编码了这些不同的可解释语义的信息是多少。这样我们就可以利用$q(z|x)$也就是我们训练好的编码器，我们给它一个输入，将图像$x$映射到$z$，并把$z$用作下游任务的特征，比如监督学习，分类任务。

#### Variational Autoencoders: Summary
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/43.png)
VAE实际上是在原来的自编码器上加入了随机成分，那么在使用VAE的时候我们不是直接取得确定的输入$x$然后获得特征$z$最后再重构$x$，而是采用随机分布和采样的思想，这样我们就能生成数据。

为了训练模型VAEs，我们定义了一个难解的密度分布，我们推导出一个下界然后优化下界，下界是变化的，“变分”指的是用近似来解决这些难解的表达式，这是模型被称为变分自动编码器的原因。

VAEs优点：VAEs就生成式模型来说是一种有据可循的方法，它使得查询推断成为可能，如此一来便能够推断出像$q(z|x)$这样的分布，这些东西对其他任务来说会是很有用的特征表征。

VAEs缺点：当我们在最大化似然函数的下界时，不像pixelRNN和pixelCNN那样明确。且它生成的数据中仍然有模糊的成分。

## Generative Adversarial Networks (GAN)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/44.png)
我们之前的PixelCNN和PixelRNN定义了一个易于处理的密度函数，通过密度函数优化训练数据的似然；VAEs有一个额外定义的隐变量$z$，有了$z$以后获得了很多的有利性质但是我们也有了一个难解的密度函数，对于该函数我们不能直接优化，我们推到了一个似然函数的下界，然后对它进行优化。

现在我们放弃显式地对密度函数建模，我们想要得到的是从分布中采样并获得质量良好的样本。GANs中不再在显式的密度函数上花费精力，而是采用一个博弈论的方法，并且模型将会习得从训练分布中生成数据，而这一实现是基于一对博弈玩家。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/45.png)
在GAN的配置中，我们真正在意的是我们想要能够从一个复杂的高维训练分布中采样。如果想从这样的分布中生成样本，是没有什么直接的来方法可以采用的（该分布十分复杂，我们无法从中采样）。我们将要采用的方法是从一个简单分布中采样，比如符合高斯分布的噪声，然后我们学习一个从这些简单分布直接到我们想要的训练分布的一个变换。那么用什么来表达这一复杂的变换？当然是神经网络。

接下来要做的是取得一些具有某一指定维度的噪声向量作为输入，然后把该向量传给一个生成器网络，之后我们要从训练分布中采样并将结果直接作为输出。对于每一个随机噪声输入，我们都想让它和来自训练分布的样本一一对应。

### Generative Adversarial Networks: Training process
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/46.png)
我们接下来训练这个网络的方式是，我们会把训练过程看做两个玩家博弈的过程，一个玩家是生成器网络（Generator），一个是判别器网络（Discriminator）。生成器网络作为玩家1会试图骗过判别器网络，欺骗的方式是生成一些看起来十分逼真的图像，同时第二个玩家，也就是判别器网络，试图把真实图片和虚假图片区别开，判别器试图正确指出哪些样本是生成器网络生成的。

我们将随机噪声输入到生成器网络，生成器网络将会生成这些图像，我们称之为来自生成器的伪样本，然后我们从训练集中取一些真实图片，我们希望判别器网络能够对每个图片样本做出正确的区分，这是真实样本还是伪样本。

我们的想法是，我们想训练一个性能良好的判别器，如果它能很好的区分真实样本和伪样本，同时如果我们的生成器能够生成一些伪造样本，而这些伪造样本能够很好的骗过判别器，那么我们就获得了一个很好的生成模型，如此一来我们将可以生成一些看起来很像训练集合中的图像的样本。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/47.png)
现在我们现在有两个玩家，需要通过一个MiniMax博弈公式联合训练这两个网络，该MiniMax目标函数就是如图所示的公式，我们的目标是让目标函数在$\theta_g$上取得最小值，$\theta_g$是指生成器网络$g$的参数；同时要在$\theta_d$上取得最大值，$\theta_d$指的是判别器网络的参数。

**公式中各项的含义**：
- 第一项是在训练数据的分布上取$\log(D(x))$的期望，$\log(D(x))$是判别器网络在输入为真实数据（训练数据）时的输出，该输出是真实数据从分布p-data中采样的似然概率；
- 第二项是对$z$取期望，$z$是从$p(z)$中采样获得的，这意味着从生成器网络中采样，同时$D(G(z))$这一项代表了以生成的伪数据为输入判别器网路的输出，也就是判别器网络对于生成网络生成的数据给出的判定结果。

**对该过程的解释**：我们的判别器的目的是最大化目标函数，也就是在$\theta_d$上取最大值，这样一来$D(x)$就会接近1，也就是使判别结果接近真，因而该值对于真实数据应该相当高，并且$D(G(z))$的值也就是判别器对伪造数据输出就会相应减小，我们希望这一值接近于0。因此如果我们能最大化这一结果，就意味着判别器能够很好的区别真实数据和伪造数据。对于生成器来说，我们希望它最小化该目标函数，也就是让$D(G(z))$接近1，如果$D(G(z))$接近1，那么用1减去它就会很小，判别器网络就会把伪造数据视为真实数据，也就意味着我们的生成器在生成真实样本。

那我们该如何训练这些网络呢。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/48.png)
这是一个无监督学习，所以不会人工给每个图片打上标签，但是生成器网络中生成的数据我们标签为0或假，我们的训练集都是真实的图片，被标记为1或真。有了这些以后，对判别器的损失函数而言就会使用这些信息，判别器要做的就是对生成器生成的图片输出0，而对真实图片输出1，这其中没有外部标签。

**如何训练**：首先对判别器进行梯度上升，从而习得$\theta_d$来最大化该目标函数。接着对生成器进行梯度下降，$\theta_g$进行梯度下降最小化目标函数，此时目标函数只取右边这一项，因为只有这一项与$\theta_g$有关。

以上就是GAN的训练方式，我们交替训练生成器和判别器，每次迭代生成器都试图骗过判别器。但在实践中，有一件事不得不注意。我们定义的生成器目标函数并不能很好地工作，看一下上图损失函数的函数空间，关于$D(G(z))$的损失函数的函数空间，$1-D(G(z))$也就是我们期望对于生成器能够最小化的项，它的函数图像如图所示，我们想要最小化该函数。但我们可以看出该损失函数越向右斜率越大，$D(G(z))$越接近1该函数的斜率越高，这意味着当我们的生成器表现很好时，我们才能获得最大梯度。另一方面，当生成器生成的样本不怎么好时，这时候的梯度相对平坦。这意味着梯度信号主要受到采样良好的区域支配，然而事实上我们想要的是从训练样本中学到知识。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/49.png)
因此，为了提高学习效率，我们接下来要做的是针对梯度定义一个不同的目标函数，去做梯度上升算法。也就是说我们不再最小化判别器正确的概率，而是进行最大化判别器出错的概率，这样就会产生一个关于最大化的目标函数，也就是最大化$\log(D(G(z)))$，可以看出函数图像是原来函数图像的反转。现在我们可以在生成样本质量还不是很好的时候获得一个很高的梯度信号。

联合训练两个网络很有挑战，而且会不稳定。交替训练的方式不可能一次训练两个网络，还有损失函数的函数空间会影响训练的动态过程。所以如何选择目标函数，从而获得更好的损失函数空间来帮助训练并使其更加平稳是一个活跃的研究方向。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/50.png)
在每一个训练迭代期都先训练判别器网络，然后训练生成器网络。
1.	对于判别器网络的k个训练步，先从噪声先验分布$z$中采样得到一个小批量样本，接着从训练数据$x$中采样获得小批量的真实样本，下面要做的将噪声样本传给生成器网络，并在生成器的输出端获得伪造图像。此时我们有了一个小批量伪造图像和小批量真实图像，我们有这些小批量数据在判别器生进行一次梯度计算，接下来利用梯度信息更新判别器参数，按照以上步骤迭代一定的次数来训练判别器。
2.	之后训练生成器，在这一步采样获得一个小批量噪声样本，将它传入生成器，对生成器进行反向传播，来优化目标函数。
3.	交替进行上述两个步骤。

### Generative Adversarial Networks: Generation process
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/51.png)
我们对生成器网络和判别器网络都进行了训练，经过几轮训练后我们就可以取得生成器网络并用它来生成新的图像。我们只需把噪声$z$传给它来生成伪造图像。

### Generative Adversarial Networks: Improvement
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/52.png)
之前GAN生成的图片分辨率低不清晰，后来出现了一些研究提升了生成样本的质量。Alex Radford提出给GANs增加卷积结构用于提升样本质量。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/53.png)
噪声向量$z$通过卷积神经网络一路变换直到输出样本。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/54.png)
本例中我们可以取得两个$z$，也就是两个不同的噪声向量，然后我们在这两个向量之间插值。这里每一行都是从随机噪声z到另一个随机噪声向量$z$的插值过程，这些都是平滑插值产生的图像。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/55.png)
我们更深入分析向量$z$的含义，可以对这些向量做一些数学运算。图中实验所做的是取得一些女性笑脸，一些淡定的女性脸，一些淡定的男性脸样本。取这三种样本的$z$向量的平均。然后取女性笑脸的平均向量减去淡定脸女性的平均向量，再加上淡定男性脸的平均向量，我们最后会得到男性的笑脸样本。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/56.png)
本例同上。这样一来我们就会看到z向量具有这样的可解释性，从而你就可以利用它来生成一些相当酷的样本。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/57.png)
2017年有大量关于GANs的研究工作。上图左边可以看到更好的训练与生成过程，即我们之前说的损失函数的改进，是的训练更稳定，这样一来就能获得几种不同结构的优质生成效果。另外我们还可以实现领域迁移和条件GANs，如上图中我们把源空间的马转换到输出空间的斑马。我们可以先获取一些马的图像并训练一个GAN，将这个GAN训练成输出和输入的马结构相同但细节特征却属于斑马。同样可以颠倒该过程，我们可以把苹果变成橘子。我们还可以通过这种方式做照片增强。上图下面的例子是场景变换，将冬天的图像转换成夏天。右上角的例子是根据文本信息生成图像。右下角是自动填色。

### Generative Adversarial Networks: Summary
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/60.png)
GANs并不使用显式的密度函数，而是利用样本来隐式表达该函数，GAN通过一种博弈的方法来训练，通过两个玩家的博弈从训练数据的分布中学会生成数据。

GANs的优点是它们可以生成目前最好的样本。

GANs的缺点是训练起来需要更多的技巧，而且训练起来比较不稳定。我们并不是直接优化一个目标函数，如果是这样的话只要后向传播就能轻松训练。事实上，我们需要努力地平衡两个网络的训练，这样就可能造成不稳定。同时我们还会由于不能够进行一些查询推断而受挫，如$p(x)$，$p(z|x)$，也就是在VAE里遇到的同样的问题。

# Summary
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_13/58.png)
