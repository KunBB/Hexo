---
title: CS231n Lecture 12 Visualizing and Understanding
date: 2018-10-07 21:30:00
categories: "CS231n"
tags:
  - Deep Leanring
  - Machine Learning
  - Artificial Intelligence
mathjax: true
---
Lecture 12主要讲解的是对卷积神经网络的可视化和解释性的研究，从网络层的特征可视化开始到基于梯度提升方法的特征匹配、特征反演，进而衍生出纹理合成、图像生成和风格转移等。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/1.png)
网络内部到底是如何运行的？是如何完成自己特定的工作的？他们需要寻找的特征类型是什么？这些中间层的作用是什么？
<!--more-->

# First Layer：Visualize Filters
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/2.png)
第一个卷积层由许多个卷积核组成，如AlexNet中有64个卷积核，每一个的size是（3,11,11）。通过卷积核在输入图像上的滑动、与图像块做点积，最后得到了第一个卷积层的输出。

由于我们得到了第一个卷积层的权重和输入图像像素的点积，我们可以通过简单地可视化得到卷积核寻找的东西，并将其看作图像显示出来。如AlexNet中的（3,11,11）的卷积核可以看做有3通道的$11\times11$的图像，并且给定红蓝绿值。因为有64个卷积核，所以我们把它看做由64个小的3通道的$11\times11$图像组成的大图像（如下图所示）。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/3.png)
我们可以看出它们都在寻找有向边（如明暗线条），从不同的角度和位置来观察输入图像，我们可以看到完全相反的颜色（如绿色和粉色，蓝色和橙色）。

无论网络结构如何，或是训练数据类型如何，第一个卷积层得到的可视化结果都与上图类似。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/4.png)
如果我们对中间层做相同的可视化操作，但实际上它的可解释性会差很多。比如第二层使用20个（16,7,7）尺寸的卷积核，我们不能直接将其转换成可视化图像。我们可以将（16,7,7）的卷积核平面展开成16个$7\times7$的灰度图像。因为这些卷积核没有直接连接到输入图像，回想一下，第二层的卷积核与第一层的输出相连接，所以这让我们意思到在第一次卷积后，什么类型的激活模式会使第二个激活层的卷积的活性最大化。但这并不是可以解释的，因为我们并不知道那些第一层的卷积在图像像素上呈现出的样子是什么样的。

# Last Layer
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/5.png)
网络的最后一层我们通过大概1000个类的得分来预测图像类别。在最后一层之前一般会有一个全连接层，以AlexNet为例，我们用4096维的特征向量来表示我们的图像，然后将其输入到最后一层来预测分类得分。

另一种用来解决可视化和理解卷积神经网络的方式是尝试去理解神经网络的最后一层到底发生了什么。我们可以提取一些数据集通过我们训练后的卷积神经网络，并为每一个图像标记对应的4096维向量，接着可视化最后一个隐层。

## Last Layer：Nearest Neighbors
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/6.png)
这里我们尝试使用最近邻算法。与之前所讲的逐像素地计算最近邻不同之处在于，这里我们计算的是最后一层卷积层所生成的4096维特征向量的最近邻。（上图左侧是ciffa-10上测试的逐像素计算最近邻，上图中部展示的是计算4096维的最近邻。）

我们可以发现这两种方法非常的不同，因为图像的像素在它的近邻和特征空间之间是非常不同的。然而这些图像的语义内容在特征空间中是相似的。例如，我们在第二行查询的是这只站在图像左侧的大象，但在其最近邻结果中，我们可以发现有站在图像右侧的大象，而它们的像素是几乎完全不同的。通过神经网络学习到的特征空间中，这两个图像彼此之间非常相似，也就是说特征空间的特性是捕捉这些图像的语义内容。

## Last Layer：Dimensionality Reduction
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/7.png)
另一个观察最后一层到底发生了什么的角度是使用降维方法。类似于PCA，有一种更强大的算法t-SNE（t分布领域嵌入）用于可视化特征的非线性降维。

这里展示了在mnist数据集上使用t-SNE降维的应用实例。我们使用t-SNE将mnist中的$28\times28$维原始像素的特征空间（4096维）压缩到2维。上图中的集群对应了mnist数据集中的数据。

于是我们在我们训练的图像分类网络中做相同类型的可视化工作。我们提取大量的图像，并且让他们在卷积神经网络上运行，记录每个图像在最后一隐层的4096维特征向量。然后我们通过t-SNE降维方法把4096维的特征空间压缩到2维特征空间。现在我们在压缩后的2维特征空间中布局网络，并且观察这个2维特征空间中网格中每个位置会出现什么类型的图像。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/8.png)
从上图我们可以发现在特征空间中有一些不连续的语义概念，与Nearest Neighbors结果类似，特征上相似的物体会聚集到一起。

# Visualizing Activations
可视化中间层的权重解释性并不是那么强，但是可视化中间层的激活映射图在某些情况下是具备可解释性的。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/9.png)
CONV5的特征向量为$128\times13\times13$，我们将其看做128个的$13\times13$的二维向量，我们可以把每一个$13\times13$的二维向量可视化为灰度图像。这可以告诉我们卷积神经网络要寻找的特征在输入中是什么类型的。

从上图可以看出大部分中间层特征都有很多的干扰，但是这里有一个突出的中间层特征（绿色方框），看起来似乎在激活对应人脸的特征映射图部分。

# Maximally Activating Patches（最大化激活块）
可视化中间层的另一种非常有用的方法是可视化输入图像中什么类型的图像块可以最大限度地激活不同的特征和不同的神经元。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/10.png)
我们选取AlexNet的卷积层，AlexNet的每一个激活量提供$128\times13\times13$的三维数据，其中每一个元素就是一个神经元，我们选取128个$13\times13$二维向量中的一个（如第17个），这里一个二维向量就是一个卷积核和输入数据点积后的运算结果。我们通过卷积神经网络运行很多图像，对于每一个图像记录它们相应部分（第17个）的卷积特征，可以观察到特征映射图的相应部分已经被我们的图像数据集最大地激活。

卷积层中的每个神经元在输入部分都有一些小的感受野，每个神经元的管辖部分并不是整个图像，它们只针对这个图像的子集合。我们要做的是从这个庞大的图像数据集中，可视化来自特定层，特定特征对应的最大激活图像块，我们可以根据这些激活块在这些特定层的激活程度来解决这个问题。

上图是一些来自激活特定神经元的输入图像块的实例，最大化激活图像块的可视化结果。我们从神经网络的每一层选择一个神将元，根据从大型数据集中提取的图像对这些神经元进行排序（根据神经元激活值大小排序），这会使这个神经元被最大程度地激活。上面是低层的，使激活的特征很明显，下面是高层的，具有更广阔的感受野，特征更高级一些。

这可以让我们了解神经元可能在寻找什么特征。如第一行中神经元可能在寻找输入图像中圆形的深色物体。

# Occlusion Experiments（遮挡实验）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/11.png)
遮挡实验的目的是弄清楚究竟是输入图像的哪个部分导致神经网络做出分类决定。我们将一幅输入图像的某个部分遮挡，将遮挡部分设置为这幅图像的平均像素值。通过神经网络运行该图像，记录遮挡图像的预测概率，然后将这个遮挡图像块划过输入图像的每个位置，并重复之前的操作，最后绘制图像的热力图，热力图显示了我们遮挡不同部位时的预测概率输出。

如果我们遮挡了图像的某个部分，并且导致了神经网络分数值的急剧变化，那么这个遮挡的输入图像部分可能对分类决策起到非常重要的作用。

# Saliency Maps（显著图）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/12.png)
给出一只狗的输入图像，以及狗的预测类标签，我们想要知道输入图像中的哪部分像素对于分类是重要的。

显著图的方法是计算输入图像像素的预测类别分数值的梯度，这将直接告诉我们在一阶近似意义上对于输入图片的每个像素如果我们进行小小的扰动，那么相应分类的分数值会有多大的变化。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/13.png)
GrabCut是一种分割算法，当将其与显著图结合起来时，我们可以细分出图像中的对象。但这种方法有一些脆弱。

# Intermediate Features via  backprop(guided)（引导式反向传播）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/14.png)
另一种方法称之为引导式反向传播。我们不使用类的分数值，而是选取神经网络中间层的一些神经元，看图像中的哪些部分影响了神经网络内的神经元的分值。仅传播正梯度，仅需跟踪整个神经网络正面积极的影响。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/15.png)
可以看出此种方法生成的图像特征更加清晰。

# Gradient Ascent（梯度上升）
神经网络相当于一个函数告诉我们输入图像的哪一部分影响了神经元的分数值，问题是如果我们移除了图像上的这种依赖性后，什么类型的输入会激活这个神经元。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/16.png)
我们现在希望通过梯度上升方法来修正训练的卷积神经网络的权重，并在图像的像素上执行梯度上升来合成图像，以尝试和最大化某些中间神经元和类的分数值。在执行梯度上升的过程中。我们不再优化，神经网络中的权重值保持不变，相反，我们试图改变一些图像的像素，使这个神经元的值或这个类的分数值最大化。除此之外，我们需要一些正则项来防止我们生成的图像过拟合特定网络的特性。

生成图像需要具备两个属性：
-	最大程度地激活一些分数值或神经元的值；
-	我们希望这个生成的图像看起来是自然的，即我们想要生成的图像具备在自然图像中的统计数据。正则项强制生成的图像看起来是自然的图像。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/17.png)
我们需要把初始图像初始化为0，或是添加高斯噪声，然后重复进行前向传播并计算当前分数值、通过反向传播计算相对于图像像素的神经元值的梯度、对图像像素执行一个小的梯度下降或上升的更新，以使神经元分数值最大化，直到我们生成了一幅漂亮的图像。

## Regularizer
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/18.png)
一个非常普通的图像正则化想法是惩罚生成图像的L2范数，这从语义上来说并不是那么有意义。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/19.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/20.png)
当你在训练的神经网络上（惩罚生成图像的l2范数）时，生成的图片如上图所示，可以看出我们正在尝试最大化左上角哑铃（dumbbell）的生成图像分数值（第一幅图）。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/21.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/22.png)
解决问题的另一个角度是通过改进正则化来改善可视化图像。除了L2范数约束外，我们还定期在优化过程中对图像进行高斯模糊处理，同时也将一些低梯度的小的像素值置0。可以看出这是一种投影梯度上升算法，定期投影具备良好属性的生成图像到更好的图像集中。例如，进行高斯模糊处理后，图像获得特殊的平滑性，更容易获得清晰的图像。

现在我们可以不仅对类的最后分数值执行这个程序，也可以对中间神经元。例如，我们可以最大化某个中间层的其中一个神经元的分数值，而不是最大化台球桌（billard table）的分数值。

以上图像都是不同随机初始化图像经过同一程序得到的结果。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/23.png)
我们可以使用这些相同类型的程序来可视化及合成图像即最大限度激活神经网络的中间神经元。然后我们就可以了解到这些中间神经元寻找的东西是什么。例如第四层可能在寻找螺旋状的东西。

一般来说，当你把图片放的越大，神经元的感受野范围也会越大，所以（使用者）应该寻找图像中较大的图像块，神经元则倾向于寻找图像中更大的结构或更复杂的类型。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/24.png)
另外在优化问题中考虑多模态问题，即对每一个类运行聚类算法以使这些类分成不同的模型，然后用接近这些模型其中之一的类进行初始化，可以得到更好的图像。

# Fooling Images / Adversarial Examples（愚弄图像/对抗样本）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/26.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/25.png)
愚弄图像，即选取一些任意的图像，比如我们提取一张大象的图像，然后告诉神经网络最大化这张图像中考拉的分数值，然后我们要做的是改变这个大象的形象，让神经网络将它归类为考拉。你可能希望的是这头大象更像一只考拉，有可爱的耳朵，但是实际上并不是如此。如果你提取这张大象的图像，然后告诉神经网络它是考拉，并且尝试改变大象的图像，你将会发现第二幅图像会被归类为考拉，但是它在我们看来和左边第一幅图是一样的。

第一列和第二列图像在像素上是没有差异的，如果你放大这些差异，并不会真的在这些差异中看到ipod或者考拉的特征，它们就像随机的噪声模式。

（为何会这样？Ian Goodfellow将会在Lecture 16：特邀讲座中讲解。）

# DeepDream
（也属于基于梯度的图像优化。）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/27.png)
提取我们输入的图像，通过神经网络运行到某一层，接着进行反向传播并且设置该层的梯度等于激活值，然后反向传播到图像并且不断更新图像。

对于以上步骤的解释：试图放大神经网络在这张图像中检测到的特征，无论那一层上存在什么样的特征，我们设置梯度等于特征值，以使神经网络放大它在图像中所检测到的特征。    这种方法同样可用于最大化图像在该层的L2范数。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/28.png)
A couple of tricks：
-	计算梯度之前抖动图像，即不是通过神经网络运行完全和原图像相同的图像，而是将图像移动两个像素，并将其他两个像素包裹其中，这是一种正则化方式，以使图像更加平滑；
-	使用梯度的L1归一化；
-	有时候修改像素值。

这是一种投影梯度下降，即投影到实际有效图像的空间上。

# Feature Inversion（特征反演）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/29.png)
选取一张图像，通过神经网络运行该图像，记录其中一个图片的特征值，然后根据它的特征表示重构那个图像，观察那个重构图像我们会发现一些在该特征向量中捕获的图像类型的信息。

我们可以通过梯度上升和正则化来做到。与其最大化某些分数值，不如最小化捕捉到的特征向量之间的距离，并且在生成图像的特征之间尝试合成一个新的与之前计算过的图像特征相匹配的图像。

一个经常见到的正则化是全变差正则化，全变差正则化将左右相邻像素间的差异拼凑成上下相邻，以尝试增加生成图像中特殊的平滑度。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/30.png)
左边是原始图像，我们通过VGG-16运行这个图像，记录这个神经网络某一层的特征，然后尝试生成一个与那层记录的特征相匹配的新图像，这让我们了解到这张图像在神经网络不同层的这些特征的信息存储量。例如当我们尝试基于VGG-16的relu2-2特征重构图像，可以看到图像被完美地重构，即不会真正丢弃该层原始像素的许多信息。但是当我们向上移动到神经网络的更深处，尝试从relu4-3、relu5-1重构图像，可以看到图像的一般空间结构被保留了下来，但是很多低层次的细节并不是原始图像真实的像素值，并且纹理的颜色也和原来不同。这些低层次的细节在神经网络的较高层更容易损失。

这让我们注意到随着图像在神经网络中层数的上升，可能会丢失图像真实像素的低层次信息，并且会试图保留图像的更多语义信息，对于类似颜色和材质的小的变化，它是不变的。

# Texture Synthesis（纹理合成）
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/31.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/32.png)
这里没有使用神经网络，而是运用了一种简单算法，即按照扫描线一次一个像素地遍历生成图像。然后根据已生成的像素查看当前像素周围的邻域，并在输入图像的图像块中计算近邻，然后从输入图像中复制像素。

上述非神经网络方法对于简单纹理生成效果很好，但是当纹理较为复杂时，可能会行不通。

## Neural Texture Synthesis: Gram Matrix
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/33.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/34.png)
首先选取我们输入的石头纹理，把它传递给卷积神经网络，然后抽取他们在神经网络中某层的卷积特征，这里我们假设卷积特征size为$C\times H\times W$，我们可以将其看做$H\times W$的空间网格，在网格上的每一点都有$C$维特征向量来描述图像在这点的外观。

我们将会使用激活映射图来计算输入纹理图像的映射符，然后选取输入特征的两个不同列，每个特征列都是$C$维的向量，然后通过这两个向量得到$C\times C$的矩阵。这个$C\times C$矩阵告诉我们两个点代表的不同特征的同现关系，如果C*C矩阵中位置索引为$ij$的元素值非常大，这意味着这两个输入向量的位置索引为$i$和$j$的元素值非常大。

这以某种方式捕获了一些二阶统计量，即映射特征图中的哪些特征倾向于在空间的不同位置一起激活。

我们将对$H\times W$网格中所有不同点所对应的特征向量取平均值，那么我们会得到$C\times C$ Gram矩阵，然后使用这些描述符来描述输入图像的纹理结构。

**关于Gram矩阵：**
它丢弃了特征体积中的所有空间信息，因为我们对图像中的每一点所对应的特征向量取平均值，它只是捕获特征间的二阶同现统计量，这最终是一个很好的纹理描述符。并且Gram矩阵的计算效率非常高，如果有$C\times H\times W$三维张量，可以对它进行重新组合得到$C\times HW$，然后乘以它本身的转置矩阵，这些计算都是一次性的。

使用协方差矩阵同样有效，但是计算成本要高。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/35.png)
 一旦我们有了纹理在神经网络上的描述符，我们可以通过梯度上升来合成与原始图像纹理相匹配的新的图像。这看起来跟我们之前说的特征重构有些类似，但是这不是试图重构输入图像的全部特征映射，而是尝试重构输入图像的Gram矩阵纹理描述符。

人们用纹理图像来训练VGG网络，并计算网络不同层的Gram矩阵，然后随机初始化新的图像，接着使用梯度上升。即随机选取一张图片，使他通过VGG，计算在各层上的Gram矩阵，并且计算输入图像纹理矩阵和生成图像纹理矩阵之间的L2范数损失，然后进行反向传播，并计算生成图像的像素值梯度，然后根据梯度上升一点点更新图像的像素，不断重复这个过程，即计算Gram矩阵的L2范数损失，反向传播图像梯度，最终会生成与纹理图像相匹配的纹理图像。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/36.png)
上图顶部是四张不同的输入纹理图像，底部是通过Gram矩阵匹配的纹理合成方法合成的新图像，即计算在预训练卷积神经网络中不同层的Gram矩阵。如果我们使用卷积神经网络的较低层，那么通常会得到颜色的斑点，总体的结构并没有被很好的保留下来。下面几行的图像，即在神经网络的较高层计算Gram矩阵，它们倾向于更大力度地重建输入图像，这在合成新图像时可以很好地匹配输入图像一般的空间统计量，但是他们在像素上和真实的输入图像本身有很大的差别。

# Neural Style Transfer
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/37.png)
如果我们将梵高的星空或其他艺术品作为输入纹理图像，然后运行相同的纹理合成算法，那么生成的图像倾向于重构那些艺术品中比较有趣的部分。

当你把Gram矩阵匹配的纹理合成方法与特征匹配的特征反演算法结合在一起，一些有趣的事情将会发生，即风格迁移。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/38.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/39.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/40.png)
风格迁移中，我们将两张图像作为输入图像。第一步，选取其中一幅图像作为内容图像，它将引导我们的生成图像看起来像什么。同样的，风格图像负责生成图像的纹理和风格。将它们输入到神经网络中以计算Gram矩阵和特征，然后使用随机噪声初始化输出图像，计算Gram矩阵的L2范数损失，以及图像上的像素梯度。不断重复上述步骤，在生成图像上执行梯度上升，以实现最小化内容图像的特征重构损失以及风格图像的Gram矩阵损失。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/41.png)
生成图像时，我们在联合重构最小化内容图像的特征重构损失和风格图像的gram矩阵损失。通过控制两个损失的权重，我们可以控制内容和风格之间在生成图像中所占的比重。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/42.png)
还有很多别的超参数，如在计算Gram矩阵前重新调整风格图像的尺寸大小，这可以让我们控制从特征图像中重构的特征的尺度。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/42.png)
还有很多别的超参数，如在计算Gram矩阵前重新调整风格图像的尺寸大小，这可以让我们控制从特征图像中重构的特征的尺度。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/43.png)
风格迁移算法有一个问题，其算法效率非常低。为了生成图像，我们需要通过预训练神经网络，计算大量的正向传播和反向传播。

![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/44.png)
解决此问题的方法是用另一个神经网络来进行风格迁移的工作。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/45.png)
即在一开始就修改我们想要迁移的风格。在这种情况下不是为我们想要合成的每个图像运行一个单独的优化程序，而是训练一个可以输入内容图像的前馈网络，直接输出风格迁移后的结果。训练前馈神将网络的方法是在训练期间计算相同内容图像和风格图像的损失，然后使用相同梯度来更新前馈神经网络的权重，一旦训练完成，只需在训练好的网络上进行单一的正向传播。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/46.png)
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/47.png)
以上两种快速风格迁移算法与第一种类似，有一些小的不同之处。

以上这些快速风格迁移算法的一个缺点是在训练新的风格迁移网络时，对于要应该的每个风格实例都需要训练一个神经网络。
![Loading...](https://raw.githubusercontent.com/KunBB/MarkdownPhotos/master/CS231n_12/48.png)
来自Google的一篇论文提出使用一个训练好的前馈神经网络对输入图像应用许多不同的风格。选取一张内容图像，以及风格图像，然后使用一个神经网络来应用许多不同类型的风格。此算法也能在一个训练好的神经网络上进行混合风格迁移。一旦你在训练这个神经网络时使用了4中不同风格的图像，你可以在测试时指定这些风格的混合。
