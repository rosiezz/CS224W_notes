@[TOC](图神经网络)
# 1 回顾
## 1.1 节点嵌入
传统机器学习难以应用在图结构上。节点嵌入是将节点映射到d维向量，使得在图中相似的节点在向量域中也相似。存在以下问题：
- 复杂度高，每个节点的嵌入向量都需要单独训练
- 无法获取在训练时没出现过节点的表示向量，无法泛化到新图或新节点
- 无法应用节点自身属性特征信息

## 1.2 GNN结构
deep graph encoders：即用GNN进行节点嵌入
![在这里插入图片描述](https://img-blog.csdnimg.cn/9db8f7db60114be3a6227c7d649dd013.png)
## 1.3 任务
- 节点分类：预测节点的标签
- 链接预测：预测两点是否相连
- 社区发现：识别密集链接的节点簇
- 网络相似性：度量图/子图间的相似性

# 2 图深度学习
**Assume:**
图 G 
节点集 V 
邻接矩阵 A（二元，无向无权图。这些内容都可以泛化到其他情况下）
节点特征矩阵 X 
一个节点 v
v 的邻居集合N(v)
如果数据集中没有节点特征，可以用指示向量indicator vectors（节点的独热编码），或者所有元素为常数1的向量。有时也会用节点度数来作为特征。

### DNN
将邻接矩阵拼和节点特征合并，用DNN训练，缺点：
- 参数多
- 如果图发生变化，邻接矩阵发生变化，无法适配原DNN（我理解是，只能用于原图）
- DNN对输入顺序比较敏感，而图是无序的，相同图不同的顺序图的邻接矩阵不一样，DNN无法处理无序的结构（我们需要一个即使改变了节点顺序，结果也不会变的模型）

---->借用CNN的思想，将网格上的卷积神经网络泛化到图上，并应用到节点特征数据。但是在CNN中，卷积核大小是固定的，而图的邻居无法用固定大小的卷积核来处理（图上无法定义固定的lacality或滑动窗口且节点顺序不固定），因此用的是**聚合（aggregation）** 思想。
### 聚合思想
1.转换邻居信息，将其加总
![在这里插入图片描述](https://img-blog.csdnimg.cn/5bb9dc15139e419e9e3705d0cae28086.png)
2. Graph Convolutional Networks
通过节点邻居定义其计算图，传播并转换信息，计算出节点表示（可以说是用邻居信息来表示一个节点）
![在这里插入图片描述](https://img-blog.csdnimg.cn/1169411bd8a64c34881acf20ff793472.png)
3.核心思想：通过聚合邻居来生成节点嵌入
通过神经网络聚合邻居信息，通过节点邻居定义计算图（它的邻居是子节点，子节点的邻居又是子节点们的子节点……）
![在这里插入图片描述](https://img-blog.csdnimg.cn/cbecceec85a448f8a7fe32f1d3b9f1b6.png)
4.深度模型：很多层
节点在每一层都有不同的表示向量，每一层节点嵌入是邻居上一层节点嵌入再加上它自己（相当于添加了自环）的聚合。
第0层是节点特征，第k层是节点通过聚合k hop邻居所形成的表示向量。
在这里就没有收敛的概念了，直接选择跑有限步（k）层。
![在这里插入图片描述](https://img-blog.csdnimg.cn/8faffda6894b43c589c0b09878b09c85.png)
5.邻居信息聚合neighborhood aggregation
不同聚合方法的区别就在于如何跨层聚合邻居节点信息。neighborhood aggregation方法必须要order invariant或者说permutation invariant
基础方法：从邻居获取信息求平均，再应用神经网络
![在这里插入图片描述](https://img-blog.csdnimg.cn/d67a2994f26c43e8bc208abcfa10a8d9.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/612e46be5a8f485084aa85190d70e9ba.png)
不同顺序下，目标节点有相同的计算图
![在这里插入图片描述](https://img-blog.csdnimg.cn/3760cd05e18b459eab66965f6f6873e5.png)
### 模型训练
模型上可以学习的参数有Wl（neighborhood aggregation的权重）和Bl转换节点自身隐藏向量的权重）（注意，每层参数在不同节点之间是共享的）。
可以通过将输出的节点表示向量输入损失函数中，运行SGD来训练参数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/d0dda0a5666e4c868c4753a74a7361b6.png)
矩阵形式
![在这里插入图片描述](https://img-blog.csdnimg.cn/7f094fea42e24832b89e30bcf36be094.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1d9a58626d784d66a265d47bff0b9088.png)
训练：监督学习和非监督学习
非监督学习
![在这里插入图片描述](https://img-blog.csdnimg.cn/297dcd1a02f249cd95f91513b8f71ab3.png)
监督学习
![在这里插入图片描述](https://img-blog.csdnimg.cn/4d60ce41ab954bfcbc83122d99c6c71c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1972417f29b342bb88e8c8a923b8f30b.png)
### 模型设计概述
第一步：定义节点聚合的函数
第二步：定义节点嵌入的损失函数
第三步：训练
第四步：生成节点嵌入
第五步：泛化到新节点
![](https://img-blog.csdnimg.cn/81a58a7da37a4e1287e0652fa9b008a2.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/17146cd82afe48c6b396bfa6998d67be.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bba1a1eb084a4f7eaa9dd29ff62e7613.png)
泛化到新节点：聚合参数在所有节点共享---->泛化到新图
![在这里插入图片描述](https://img-blog.csdnimg.cn/6e101e5cddba45d6bfbc4bb8833d71d7.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c560ae27907d40e6b996e245d8baa498.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/dab7a44f4dbc4b8d8b698fcf8ed6f822.png)
GNN CNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/c4693dae4f314ac8a799d72d7830a36a.png)
CNN可以看作时一个有固定邻域和固定顺序特殊的GNN
- CNN卷积核尺寸是预先定义的
- GNN的优势在于可以处理任意一个节点度不同的图
- CNN不是顺序不变的，改变像素的顺序可以得到不同的输出

GNN transformer
transformer:处理顺序问题，核心是自注意力
- transformer可以看作是一个全连接词图的特殊GNN

![在这里插入图片描述](https://img-blog.csdnimg.cn/f1c6a30c27d94347a58db39e123e28fe.png)

# 3 图神经网络
## 3.1 GNN 基础
### 3.1.1 框架
- 原始输入图=!计算图（特征扩增和结构扩增）
- GNN layer：邻域信息转换+邻域信息聚合
- layer connectivity:按顺序堆叠
- 训练：监督或半监督
- 训练目标：节点、边、图
![在这里插入图片描述](https://img-blog.csdnimg.cn/5e3ca22d3d75422ca2c5767215500dd5.png)
### 3.1.2 GNN layer=邻域信息转换+邻域信息聚合
1. 有多种例子，GCN/ GraphSAGE/GAT
- 把多个邻居节点和自己的向量转换成一个向量
- 邻居节点的向量集合与节点顺序无关

2. 邻域信息转换
![在这里插入图片描述](https://img-blog.csdnimg.cn/d5ad3b4dd65e429bba08f97b14171f6f.png)
2. 邻域信息聚合
聚合方法可以是求和、平均、最大
![在这里插入图片描述](https://img-blog.csdnimg.cn/b5daf766dacc4d858fd961150b41b07c.png)
3. 存在问题
如果只用邻居节点的信息，说明该节点只取决于邻居节点---->解决办法：加上自己的节点信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/7905cb9ed75544c182d28564b55f77cb.png)
4. GNN layer
![在这里插入图片描述](https://img-blog.csdnimg.cn/e96d18d5cf9f4c3a941a36b1c11db1f0.png)
5. Classical GNN layer (GCN)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c33824d9cf4a4e959d87b3362395acdd.png)
6. GraphSAGE
个人理解是用到了邻域信息和自己的信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/2e27e27e7f124e72a47a3f28455330f3.png)
AGG的方法有
![在这里插入图片描述](https://img-blog.csdnimg.cn/096af81a25134b4a9c6ca52b8369d52c.png)
l2归一化
![在这里插入图片描述](https://img-blog.csdnimg.cn/1f9d9145b0ff418296012b2a50c92544.png)
7. GAT graph attention networks
在 GCN和GraphSAGE中，邻居节点对该节点的权重系数取决于图的连接结构（节点度），且不同邻居带来的信息权重相同
--->实际上，不同节点带来的权重是不同的，引入注意力权重
第一步：用自注意力函数对每个邻居节点计算一个注意力分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/a53c79d6fd664c1abddc3230be353eff.png)
第二步：自注意力函数
- 自注意力系数是一个标量
- 自注意力函数自定义，可以是一个单层神经网络
![在这里插入图片描述](https://img-blog.csdnimg.cn/950f5de342ab467dbd946bf8ea5d7453.png)
第三步：用softmax 将自注意力系数归一化为权重，最后加权求和得到聚合结果![在这里插入图片描述](https://img-blog.csdnimg.cn/ec6110cd4c7f4383b0c635efad75560a.png)
注意力机制改进---多头注意力机制：避免偏见，陷入局部最优
分别训练不同的自注意力函数，每个函数对应一套自注意力权重
![在这里插入图片描述](https://img-blog.csdnimg.cn/293452d8446d4baca2b5a842356d8a82.png)
自注意力机制的优点：
- 不同节点权重不同
- 计算高效：可以并行计算
- 存储高效
- 局部图参与计算
- 泛化
![在这里插入图片描述](https://img-blog.csdnimg.cn/b441886fe2bb4aecb5ea04ebcfb46268.png)
### 3.1.3 GNN layer in practice
1. 框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/4d0852840ac0416983eb778bb0a7a671.png)
2. batch normalization:标准化
![在这里插入图片描述](https://img-blog.csdnimg.cn/ec7e97544f1b489c938424165122bc3b.png)
3. dropout:防止过拟合
![在这里插入图片描述](https://img-blog.csdnimg.cn/78e8d7bfc54f4c2d8002bdb783ce465b.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f005f48725da403c93cd2f2a54fc9788.png)4. activation 非线性
![在这里插入图片描述](https://img-blog.csdnimg.cn/904f60a7725242e18d831baee2d7cfba.png)
5. 小结![在这里插入图片描述](https://img-blog.csdnimg.cn/051b4ab03983411087fc31836a41b68e.png)
### 3.1.4 GNN layers stacking
- 按顺序堆叠layers
- 输入：原始节点属性特征，输出：节点嵌入
![在这里插入图片描述](https://img-blog.csdnimg.cn/60f8698acda54b699ce91f435f9ade53.png)
1. 过度平滑问题
GNN层数不能过深
- 所有节点的嵌入相同

K层GNN的感受野：决定节点嵌入的节点集合
- 共享邻居节点随着GNN层数的增加快速增加
![在这里插入图片描述](https://img-blog.csdnimg.cn/c74e8d2da4a940fbb0c70efbf2afc51f.png)
一个节点的嵌入取决于它的感受野，如果两个节点有高度重合的感受野，那么他们的嵌入就会非常相似
---->出现过度平滑的问题
解决方法：从GNNlayers的连接入手
（1）不能无脑堆很多层
第一步：计算必要的感受野（例如计算图的直径）
第二步：将GNNlayers的个数稍稍大于我们需要的感受野，可以用automl找到最优的层数
 
 如果GNN层数很小，如果提高GNN的表示能力？
 1）在GNN层的内部加上深度神经网络
 在之前的例子中，邻域信息转换或聚合只有一个线性层
 2）预处理和后处理
 在GNN层的前后加MLP层
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/b28f48a8d9d24b0898da1a6561fcb0e1.png)
如果需要很多GNN 层呢？
解决方法（2）在GNN中增加skip connections
GNN靠前层（感受野较小的层）的节点嵌入能够较好的区分节点--->通过裁剪增加前面层的影响
![在这里插入图片描述](https://img-blog.csdnimg.cn/2aacb70fdeca428eaf515dc83a484cff.png)
skip connections可以创造混合模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/a4691a55d5de40999b9f8570d0a7e87c.png)
例子:GCN with skip conections
![](https://img-blog.csdnimg.cn/1c546844b7c14e2abf4acd3e6f8d88fc.png)
其他例子;
![在这里插入图片描述](https://img-blog.csdnimg.cn/2e170053f9dc4c8d83d2f8f0eccf7d5c.png)
### 3.1.5 输入图和特征的扩增
在这之前，都是假设原始图==计算图
但是，
- 节点特征层面，输入图缺少节点特征--->特征增强
- 图结构层面
	* 太过稀疏---消息传递效率低--->添加虚拟节点和边
	* 太过稠密---消息传递耗资源--->消息传递时对邻居节点采样
	* 太大---计算图与GPU不匹配--->计算嵌入时采样子图
- 原始图 不太可能 恰好就是最优的计算图

1. 为什么要进行特征增强？
（1）正常情况下我们只有邻接矩阵，没有节点属性
（a)给节点分配一个常数，也可以是固定长度的常数向量
（b)给节点分配一个唯一ID编号，这些ID编号转化成独热向量
![在这里插入图片描述](https://img-blog.csdnimg.cn/bc4f4cb140d044a4b232197b135243d0.png)
（2）用GNN很难学习一些图结构
例如：数节点所在环的长度，GNN学不到
- 因为所有节点度相同（除非有属性特征）
- 计算图是相同的二叉树
![在这里插入图片描述](https://img-blog.csdnimg.cn/e11a6d886843472290d94b7f89556289.png)
解决方法：人为补充信息至节点属性特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/b732d7a64730461f884d9f3308a8fa0e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/26823abafdf446f6a039325564848b90.png)
2. 添加虚拟节点和边
目的：增强稀疏图
（1）虚拟边
![在这里插入图片描述](https://img-blog.csdnimg.cn/b30dd444f7f44be29637ed68fcb30417.png)
（2）虚拟节点
![在这里插入图片描述](https://img-blog.csdnimg.cn/d4a28689c02846a1a36c2a2fb1fcf0ff.png)
3. 消息传递时对邻居节点采样
目的：稠密图（所有节点用来传递消息）
该方法是对选取部分节点用来传递信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/a0fa24b1387b4f52bc69b924be5d8454.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f167a47f22fe403faf4930f97474b235.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b6a163d46a6c471e9e21c0e976e7602e.png)
### 3.1.6 GNN训练--rediction head
以上步骤可以得到节点嵌入，下一步是prediction head
- 节点层面
- 边层面
- 图层面
不同的预测任务需要不同的prediction head
![在这里插入图片描述](https://img-blog.csdnimg.cn/ca851baad097440cbec347d36868a8bf.png)
(1) 节点层面
经过GNN计算，得到了节点的D维向量，可以直接使用节点嵌入进行预测
可以进行分类（k个类别的概率）或者回归（k个连续值）
![在这里插入图片描述](https://img-blog.csdnimg.cn/cebbd867793a43958efee56143a1bff5.png)
（2）边层面
用节点嵌入对进行预测
![在这里插入图片描述](https://img-blog.csdnimg.cn/c5d1be8f0af94762b4df8702d5eca1f9.png)
（a)联结+线性
![在这里插入图片描述](https://img-blog.csdnimg.cn/33909d7f2112497d846505327c2ffbcc.png)
（2）点乘
![在这里插入图片描述](https://img-blog.csdnimg.cn/b78c8cf6ef474c7ebb3e291b6e6bc515.png)
（3）图层面
用所有节点的嵌入进行预测
![在这里插入图片描述](https://img-blog.csdnimg.cn/1ecd890b4d6d4f03bcf139effccb490c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/88533e1b578f4cf685cc092f485fc334.png)

以上方法在小图中效果很好，在大图中会丢失信息，在大型图中怎么做呢？
比如在下面的例子中，两个图的预测结果相同。
![在这里插入图片描述](https://img-blog.csdnimg.cn/72875abeabd844fe9fdad413b7496c8b.png)
--->解决方法：分层聚合节点嵌入
![在这里插入图片描述](https://img-blog.csdnimg.cn/024e93f58a3849a6ae01bee3b0410bae.png)
示例1：社群分层池化
a.分层池化
b.利用2个独立的GNN，A计算节点嵌入，B计算节点聚类
c.两个GNN可以同时执行

对于每个池化层
  * 使用GNN B的聚类结果来聚合GNN A生成的节点嵌入
  * 为每个集群创建一个新节点，保持集群之间的边缘以生成一个新的池网络
联合训练GNN A和GNN B
![在这里插入图片描述](https://img-blog.csdnimg.cn/a12e835b7e7b4fb0b99651b502aef5a1.png)
### 3.1.7 GNN 训练---预测和标签
1. 真实值来自监督学习中的标签或者无监督学习的信号
![在这里插入图片描述](https://img-blog.csdnimg.cn/0eecce0e02bc4bb79acaa7b8c8c56fa4.png)
2. 监督学习和无监督学习
blurry模糊不清的
![在这里插入图片描述](https://img-blog.csdnimg.cn/75d1d2bd8eb8406ebdb31938faafae6e.png)
监督学习标签，不同案例标签不同
![在这里插入图片描述](https://img-blog.csdnimg.cn/c5cb6e942559426e988809d9c3f79068.png)
无监督学习
![在这里插入图片描述](https://img-blog.csdnimg.cn/1e5e7c4bcd914400a3cc1ea07d244c8e.png)
### 3. 1.8 GNN训练---损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/4beefdb09f144c65a51b0a4218e3c20c.png)
分类，结果为离散值；回归，结果为连续值。GNN可以应用于这两种情况。
不同之处在于损失函数和评价指标

1. 分类的损失函数
交叉熵
![在这里插入图片描述](https://img-blog.csdnimg.cn/f41fc40cad9c49de8ea25f62e25e1870.png)
2. 回归的损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/5d59a9651881441b984394e95b3feeeb.png)
### 3.1.9 GNN 训练--评价指标
精确度和混淆矩阵，可以用sklearn实现
1. 回归
![在这里插入图片描述](https://img-blog.csdnimg.cn/e88acb6081674ef594228255c57feaeb.png)
2. 分类
![在这里插入图片描述](https://img-blog.csdnimg.cn/b26fb74ab9234da389222dbe9ab518de.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fa5d7454efa441c2bf5de3d7c0e4736e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/19e24db9a06b4f09bc96e44ccff87cb2.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/a56c5de1e2404e7b869c8a18e26646ad.png)
### 3.1.10 建立GNN 预测任务
如果分配训练集、验证集、测试集
1. 数据集分配方法： 固定分配、随机分配
![在这里插入图片描述](https://img-blog.csdnimg.cn/59d5029b13d941e9a2d6590c4b2f09eb.png)
对于图像分类，每个图像是一个数据，样本之间独立同分布
但是对于节点分类，节点之间并不是独立同分布的，所以训练集节点的计算图中可能出现测试集节点
2. GNN 中如何分配？
（1）直推式学习
输入图在训练、验证和测试时都可以用，只分配标签
训练时，用整图计算节点嵌入，训练时采用1、2节点的标签
验证时，用整图计算节点嵌入，采用3、4节点的标签评价
![在这里插入图片描述](https://img-blog.csdnimg.cn/9eadc90637174e8e82120c379f59b5c9.png)
（2）归纳式学习
将分配集之间的边打断，得到多个子图，得到的三个图时相互独立的
训练时，用节点1、2计算嵌入，节点1、2训练
验证时，用3、4计算嵌入，3、4评价

比较：
直推式学习，训练验证和测试在同一个图上，用节点标签分配，只能用于节点预测和边预测任务
归纳式学习，训练验证和测试在不同的图上，可以用于节点预测、边预测和图任务中。好的模型应该可以泛化到新图


例子：
a.节点分类：直推式和归纳式
b.图分类：归纳式
![在这里插入图片描述](https://img-blog.csdnimg.cn/c4083ac9047847b497c604c23142a753.png)
c.边预测：预测缺失的边
一个无监督/自我监督的任务。我们需要自己创建标签和数据集
具体来说，我们需要对GNN隐藏一些边，并让GNN预测这些边是否存在
第一步：创建数据集，一部分边作为message edges 一部分作为supervision edges(应该隐藏起来）
![在这里插入图片描述](https://img-blog.csdnimg.cn/cef93f2bf5ff4d6c8a792274bf5bd40d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/9d1ce526f8244ca7b5cdcda6ea13c8d7.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8500110a95ed4e339c36feb691734334.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7a5749694ec6456d84d92853c86b84d8.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/492f7968878740bab9ec8459d728d8f3.png)
## 3.2 GNN的表示能力
### 3.2.1 GNN基础概念
1. 单个GNN layer
信息计算：$m_u^{l}   =MSG^l \ (h_u^{l-1})$
信息聚合：$h_u^{l}   =AGG^l \ (m_u^{l-1},u\in\N(v))$

2. GNN model
GCN 对应元素求均值+线性+ReLU非线性
GraphSAGE 多层感知器+d对应元素max-pooling
![在这里插入图片描述](https://img-blog.csdnimg.cn/c49f78688d2f4d0192f465cc61853273.png)
- 可以用不同颜色表示不同的节点嵌入，例如表示节点的属性特征，
- 计算图中通过连接结构区别不同节点

### 3.2.2 GNN如何区分不同图结构
1.  局部邻域结构
- 如果只考虑节点的局部邻域结构
节点度不同，邻居的节点度不同
但是当两节点对称时，仅通过图的连接结构是无法区分的，如下图的节点1、2
![在这里插入图片描述](https://img-blog.csdnimg.cn/efcb7a22c31a48b5af9d363d1cf7e807.png)
2. 关键问题：GNN节点嵌入是都能够区分不同节点的局部邻域结构？

（1）从计算图入手，GNN的表达能力=区分计算图根节点嵌入的能力
在每一层，GNN聚合了邻居节点的嵌入信息，但是GNN 不关心节点编号，它只是聚合了不同节点的特征向量。
不同局部邻域定义不同的计算图，计算图与每个节点周围的根节点子树结构相同
![在这里插入图片描述](https://img-blog.csdnimg.cn/cbd2fbf4a0aa40929fbf4a6b86aec9a4.png)
（2）引入单射函数（每个输入对应唯一输出，包含了所有的输入信息）
将不同的根子树映射到不同的节点嵌入中，先获得树的单个级别的结构，然后利用递归算法得到树的整个结构
（3）如果GNN聚合的每一步都可以完全保留相邻信息，则生成的节点嵌入可以区分不同的有根子树
（4）理想GNN：不同的计算图根节点输出不同的Embedding
最具有表达力的GNN：聚合操作应该单射
最完美的单射聚合操作：哈希
### 3.2.3 设计最具表现力的GNN
1. 首先：
- GNN的表达能力可以通过使用邻域聚合函数来表征
- 更具表现力的聚合函数会使得GNN更具表现力
- 单射聚合函数可以得到最具表现力的GNN

2. 理论分析聚合函数的表达力
邻居聚合可以概括为一个超集（multi-set,一个有重复元素的集合)函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/ae2adee4d2354a46b2c2d5220a00a80f.png)

3. 分析两个GNN模型的聚合函数
- GCN的聚合函数（平均池化层）：逐元素求平均+线性+ReLU激活
无法区分颜色比例相同的不同超集，如下图（比例相同，求均值激活后输出相同）
![在这里插入图片描述](https://img-blog.csdnimg.cn/c3fb2842b69341968a5087de48c13393.png)
- GraphSAGE的聚合函数无法区分不同的超集与同一集合下的不同颜色，逐元素求最大值
![在这里插入图片描述](https://img-blog.csdnimg.cn/ed290a7562dd41b3aaa668177543861c.png)
4. 小结
- GNN的表达能力可以用邻居聚合函数的表达能力来表征。
- 邻居聚合是一个多集(具有重复元素的集合)上的函数
- GCN和GraphSAGE的聚合函数不能区分一些基本的多集;不是单射。
- GCN和GraphSAGE并不是最强大的GNN
### 3.2.4 设计表达力最强的GNN---GIN
1. 目标：
在信息传递GNN设计表达力最强的GNN，通过在多重集合上设计单射邻域聚合函数来实现
--->用神经网络拟合单射函数（神经网络的万能近似定理）

2. 任何单射多集函数都可以表示为 $\Phi(\sum\limits_{x\in\ S}f(x))$
![在这里插入图片描述](https://img-blog.csdnimg.cn/92b9d671f8d640b5bf6addc3ab5f7108.png)
例子：f产生颜色，求和记录颜色个数，Φ是单射函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/07f9ffb0bee3432d9f4034251c882c28.png)
3. 如何建立Φ和f呢？GIN网络
- 万能近似定理，使用多层感知机
在具有一个隐藏层的MLP模型中，当隐藏层的维度足够大，且使用非线性函数可以将任何连续函数近似到任意精度

- $MLP_\Phi(\sum\limits_{x\in\ S}MLP_f(x))$
实际中，MLP隐藏层的维度100-500是足够的

4. GIN网络
GIN 与 WL 图核（传统提取图级特征的方法见L3）
算法步骤：
第一步：初始化每个节点的颜色$c^{(0)}(v)$
第二步：对节点的颜色进行迭代更新 $c^{(k+1)}(v)=HASH(c^{(k)}(v),{c^{(k)}(u)}_{u\in\ N(v)})$
![在这里插入图片描述](https://img-blog.csdnimg.cn/4bdc52d209ca46f7901c5c48d3eed043.png)

第三步：迭代停止得到节点 颜色

任何一个单射函数可以表述为以下形式：$MLP_\Phi((1+\epsilon)MLP_f(c^{(k)}(v)+\sum\limits_{u\in\ N(v)}MLP_f(c^{(k)}(u))$

如果输入特征$c^{(0)}(v)$表示为独热向量，那么直接求和的函数就是单射函数（不需要f)
单射函数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8f2b7cda533c443e9b8211626444a740.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a587e4d75364a0882c9d38c6dc73562.png)
5. GIN 与 WL graph kernal
GIN可以理解为WL graph Kernel的可微神经版本
![在这里插入图片描述](https://img-blog.csdnimg.cn/819946a862b4486793df26ec424eb7aa.png)
GIN相对于WL图内核的优点：
a.节点嵌入是低维的；因此，它们可以捕获不同节点的细粒度相似性
b.可以根据下游任务学习优化

6. 总结
WL是表达能力的上界，如果两个图可以用GIN区分，那么也可以被WL kernal.反之亦然
WL kernal在理论上和经验上可以区分真实世界的大部分图
GIN也足够强大，可以区分大多数真实的图!

### 3.2.5 问题
有循环的依然不能区分
![在这里插入图片描述](https://img-blog.csdnimg.cn/5fcfd9766b834d6ebb330f174352d1e5.png)
### 3.2.6 问题与解决方案
1. 通用解决方案
* 数据预处理：特征标准化处理
* 优化器：使用ADAM优化器
* 激活函数：ReLU通常效果很良好，可使用LeakyReLU、PReLU等其他激活函数，输出层没有激活函数，每一层包含偏置项
* 嵌入维度：通常选择32、64和128

2. 调试深度网络
- Debug问题：损失/准确值在训练时未收敛
  * 检查pipeline
  * 调整学习率等超参数
  * 注意权重参数初始化
  * 仔细观察损失函数
- 模型开发：
  * 在训练集上存在过拟合情况
  用一个小的训练数据集，损失应该基本上接近于O，具有表达性的神经网络
  * 检查loss曲线
 
### 3.2.7 Resources on GNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/c5b9931e23434e029614bc9d9ae5dd7d.png)















