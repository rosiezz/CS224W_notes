# 1 PageRank概述
## 1.1 主要贡献
> **最大的贡献**：把互联网当作一个整体的系统，用有向图表示，网页之间存在关联。
>每个网页是一个节点，网页之间的引用就是连接
>pagerank就是在计算每个节点的重要度

## 1.2 互联网表示对比
目前节点或者网页时随时生成的，有些内容是私密的
由于以上两点，现在的互联网已无法直接表示成这种图
![在这里插入图片描述](https://img-blog.csdnimg.cn/f96a6bea4b91485a8c0719fad44feaa2.png)
早年，网页链接是导航链接式网页
现在，网页是交互
![在这里插入图片描述](https://img-blog.csdnimg.cn/f00227fdff524bcdab3bb1979632cca1.png)
## 1.3 实现步骤
1.构建互联网图
![在这里插入图片描述](https://img-blog.csdnimg.cn/60d88b03bebd4b3da0771ae2a7126670.png)
2.计算重要度：
使用**连接信息**定量计算节点重要度
互联网上每个网页的重要度是不相同的，符合二八定律、长尾分布或幂律分布，这种网络称为**无标度网络**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/157332199ee841f3b5e0239eec063595.png)
PageRank：计算节点重要度
Personalized PageRank：计算节点相似度，可以用于推荐系统
Random Walk with Restarts：计算节点相似度

# 2 Links as votes
以**in-coming links**为投票（以进入这个节点的链接投票，比较客观）
**link之间的重要度也是不同**的：如斯坦福官网的引用，有的是大佬，有的是小白
因此，如果引用来自重要的网页，该引用的投票权重也就更高===**递归问题**
![在这里插入图片描述](https://img-blog.csdnimg.cn/e3e3c4e76071479483c3f64f3b21e867.png)
从五个角度理解
## 2.1 迭代求解线性方程组
pagej只关注它的入节点，每个入节点i对节点j的投票权重为ri/节点j的出度
![在这里插入图片描述](https://img-blog.csdnimg.cn/8cf686d3a36144f685bf139a3db02aea.png)
举例，联立方程组，所有节点的pagerank求和为1。
该方法不推荐，扩展性差
![在这里插入图片描述](https://img-blog.csdnimg.cn/dd33fe9ade8e41e3bd30d92999834b6c.png)
初始化所有节点Pagerank，迭代求解
![在这里插入图片描述](https://img-blog.csdnimg.cn/e8c6afb3620e42d99bfb146202cb37ec.png)
## 2.2 迭代左乘M矩阵
![在这里插入图片描述](https://img-blog.csdnimg.cn/204e0244ff73400c91e71c0b00cde0c0.png)
M矩阵，每一列求和为1。也需要迭代求解。
与线性方程组等价
![在这里插入图片描述](https://img-blog.csdnimg.cn/5e58eb0445f3411f85e046b518ef30fd.png)
## 2.3 矩阵的特征向量以及特征值
可以理解为
对于一个向量左乘一个A矩阵==一个标量乘以该向量
左乘A矩阵相当于对该向量进行一次线性变换，如果线性变换后长度缩放为原来的λ倍，则该向量为矩阵A的特征向量，λ为特征值
![在这里插入图片描述](https://img-blog.csdnimg.cn/449cff15a7ac4029a02e9deca24ab2ea.png)
r向量是M的特征向量，1为特征值。
不断迭代，相当于不断地左乘M矩阵，最后是一个稳定值。这个稳定值就是M地主特征向量。
不断地左乘M矩阵称作幂迭代。
![在这里插入图片描述](https://img-blog.csdnimg.cn/b7e08ef23ca742bc80e771992166b380.png)
根据该定理，对于这个矩阵而言，PageRank是可以求解的且可以收敛
![在这里插入图片描述](https://img-blog.csdnimg.cn/288a3555ca504505ba3c3085bd7feb6a.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7c35d227c4e24d8d9e2ab2936065b71f.png)
## 2.4 Random Walk
随机游走，经过次数越多，节点越重要，归一化成概率，就得到pagerank值
![在这里插入图片描述](https://img-blog.csdnimg.cn/c98f67beae5b43a28a3e9132cc8db6ba.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e75b7f7f6cdc44629e13e59dcfd713ef.png)
## 2.5 马尔科夫链
根据状态转移矩阵计算下一时刻可能的状态的概率==随机游走
等价于求马尔科夫链的稳定解
## 2.6 总结
![在这里插入图片描述](https://img-blog.csdnimg.cn/18061a71bcc24ae7a5062302afcf5764.png)
# 3 求解
## 3.1 基本步骤

 1. 初始化pagerank值 
 2. 迭代左乘M矩阵， 
 3. 直至与上一次迭代的pagerank值小于某一阈值，（这个差值可以自定义，比如欧式距离）
 4. 大约50次迭代

![在这里插入图片描述](https://img-blog.csdnimg.cn/b97941d50c074c9192c25b48014ef9af.png)
## 3.2 收敛性分析
![在这里插入图片描述](https://img-blog.csdnimg.cn/fb1b6ff0f2574c7cb55c8936d3a158a6.png)
### 3.2.1 是否能收敛至稳定值
![在这里插入图片描述](https://img-blog.csdnimg.cn/de499be987eb4638a51142b4eb165180.png)
马尔科夫链是不可约的。互联网基本上所有节点之间是互通的，满足该条件。
直观上，一个不可约的马尔可夫链，从任意状态出发，当经过充分长时间后，可以到达任意状态。
![在这里插入图片描述](https://img-blog.csdnimg.cn/234c61d66d2b4f839c2cc7e57f00ec82.png)

周期性震荡的马尔科夫链。周期性互相串门。如下图。显然，互联网也不是这种马尔科夫链
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe65bc9c3c024a3cb2fabd09bc8e1132.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/0b56e3c1c32e4d89bbde5c12d15fd996.png)
**结论：有唯一的稳定解，且所有初始解都可以稳定到稳定解**
## 3.2 不同初始值，是否收敛至同一个结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/5fb6f3303fb841b1a02d96560491472e.png)

## 3.3 收敛的结果在数学上是否有意义&是否代表节点重要度
对于以下两种情况，即使收敛，得到的结果不是我们想要的节点重要度的值

 - [ ] **这种情况下的马尔可夫矩阵是可约的？**

![在这里插入图片描述](https://img-blog.csdnimg.cn/3740a69474e34fd6810b85f17788a1cc.png)
仅指向自己的节点：
**~~数学上没有问题~~ **，特征值，特征向量仍存在，仍能收敛，但PageRank值无意义
![在这里插入图片描述](https://img-blog.csdnimg.cn/a93fd10c6f784a5b824d6eb5825b2238.png)
如何解决？
下一时刻有β概率随机游走
1-β的概率被随机分配到其他节点
β值为0.8~0.9
![在这里插入图片描述](https://img-blog.csdnimg.cn/0b9879a3f0bc4ab8af117779e519b32d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/583ae19d89564cd1a75efb60713d9882.png)

没有出度，死胡同
数学角度存在问题，不收敛
![在这里插入图片描述](https://img-blog.csdnimg.cn/4e7c995cb3034b118ecfb8753bacf3d2.png)
如何解决？：百分百的概率被随机分配到其他节点
多个彼此的连通域：
![在这里插入图片描述](https://img-blog.csdnimg.cn/aa51bef2d96b466093c8e3acdd70a06c.png)
总结：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3594ae04d9324aa6be335551ac83c5a2.png)
## 3.4 有趣的例子
![在这里插入图片描述](https://img-blog.csdnimg.cn/7a5e97faf7654c2eae58f9fcb0bbd78a.png)
后宫干政：节点C
pagerank不能自己刷高，需要找重要的网页引用自己。
## 3.5 PageRank变种
### 3.5.1 MapReduce
并行计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/3796a966513b4a5480c17d4b63ad405c.png)
### 3.5.2  计算节点相似度
举例：bipartite graph 推荐系统
两类节点：商品和用户
目标：寻找与指定节点最相似的节点
![在这里插入图片描述](https://img-blog.csdnimg.cn/96687a1e880c4c14a074193fe0f301c1.png)
需要一个指标定量衡量节点相似度
![在这里插入图片描述](https://img-blog.csdnimg.cn/56e3c71618d64cefa77a666138c267e5.png)
方法：随机游走
query_nodes指定节点集，可以是所有节点，可以是部分节点，可以是一个节点
![在这里插入图片描述](https://img-blog.csdnimg.cn/b7ade24f5544425b889aea33837c67bc.png)
假设计算与Q相似的节点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2c99e45f5ddb48e481a57e9bc19ba0d7.png)
每次访问的时候都随机游走，并以α概率返回到指定节点集的某一点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a93164bae1b641cda5d8a6a846635ee8.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe8f91099f8e4475823f0e6f6812c790.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/531038b19be54cc495f5e9c81a8b8ebb.png)
# 4 总结
![在这里插入图片描述](https://img-blog.csdnimg.cn/347483a6530e415cbc28ea6f4e7f24eb.png)
# 5 思考题
![在这里插入图片描述](https://img-blog.csdnimg.cn/48616170b99e4c86939ac49b66501c30.png)

 1. 不可以
 2. 因为不能恶意提高自己的PageRank值，需要自己提高实力
 4.个人认为不合理
 10.不科学，可以利用节点相似度对不同节点设置不同的概率值 ，也就是与该节点相似的节点被传到该节点的概率比较高。

