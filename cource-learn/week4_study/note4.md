# 神经网络与文本处理（DL与NLP）

## 神经网络的基本概念

### 1. 生物神经元与人工神经元的类比

神经网络的灵感来源于人类大脑的工作方式，但它并不是对大脑的完美复制，而是一种数学上的抽象和模拟。

#### 生物神经元（Biological Neuron）

在生物学中，一个神经元主要由以下部分组成：

- **树突 (Dendrites)**：接收来自其他神经元的信号（输入）。
- **细胞体 (Cell Body/Soma)**：处理信号，判断信号强度是否超过阈值。
- **轴突 (Axon)**：如果信号足够强，通过轴突将信号传递给下一个神经元（输出）。

#### 人工神经元（Artificial Neuron / Perceptron）

为了模拟这个过程，科学家设计了**感知器（Perceptron）**，它是神经网络最小的运作单位：

1. **输入 (Inputs,** ![img](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg)**)**：对应“树突”，接收数据（如图片的像素值）。
2. **权重 (Weights,** ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg)**)**：对应“突触的连接强度”。有些输入很重要（权重高），有些不重要（权重低）。

- - *类比*：做决定“今晚吃火锅吗？”朋友的建议（输入）很重要，权重大；路人的建议不重要，权重小。

1. **加权求和 (Weighted Sum,** ![img](https://cdn.nlark.com/yuque/__latex/6bec5bf43c9924bf5006e907b643f00c.svg)**)**：对应“细胞体”的汇聚作用。将所有输入乘以对应的权重后相加。
2. **偏置 (Bias,** ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)**)**：一个额外的阈值调整参数，保证神经元在输入为0时也能被激活（类似函数的截距）。
3. **激活函数 (Activation Function)**：**核心概念！**

- - **作用**：决定神经元是否“开火”（Fire）。
  - 它引入了**非线性因素**，让神经网络能够解决复杂问题（不仅仅是画一条直线）。
  - *通俗类比*：像水龙头的开关。只有水压（加权和）达到一定程度，或者经过开关（激活函数）的调节，水（信号）才能流向下一层。

------

### 2. 神经网络的基本架构（层次结构）

当成千上万个“人工神经元”连接在一起时，就构成了神经网络。它有着非常严格的层级结构（Layered Structure）。

#### 三大核心层

1. **输入层 (Input Layer)**：

- - 网络的“眼睛”和“耳朵”。
  - 负责接收原始数据（如文本的词向量、图片的像素矩阵）。
  - **注意**：这一层**不进行任何计算**，只负责传输数据。

1. **隐藏层 (Hidden Layer)**：

- - 网络的“大脑”或“黑盒”。
  - 夹在输入和输出之间，可以有一层，也可以有上百层（这就是“深度”学习的由来）。
  - 负责**特征提取**。每一层都在从原始数据中提取更抽象的特征（例如：第一层识别线条，第二层识别形状，第三层识别物体）。
  - *同层限制*：在标准的前馈神经网络（Feedforward NN）中，**同一层的神经元之间互不连接，无法进行信息交换**。

1. **输出层 (Output Layer)**：

- - 网络的“嘴巴”。
  - 输出最终的预测结果（如：“这是猫的概率是90%”）。

#### 信号流向：前向传播 (Forward Propagation)

- **单向通行**：信号只能从输入层 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 隐藏层 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 输出层。
- *类比*：像工厂的流水线。原材料（数据）进厂，经过中间各个车间（隐藏层）的加工处理，最后变成成品（结果）出厂。不能逆向把成品变成原材料。

![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614542650-40d46480-e295-4aea-98c9-3ef9ca51331e.png)

------

### 3. 为什么深度学习(DL)比机器学习(ML)更复杂？

#### 核心区别：特征提取的方式

- **传统机器学习 (ML)**：

- - **人工特征工程 (Hand-crafted Features)**：需要人类专家先告诉计算机“看哪里”。
  - *案例*：识别猫。ML需要人先定义“猫有圆耳朵”、“猫有胡须”。如果定义的特征不好，模型就废了。
  - **计算模式**：虽然也有矩阵，但更多依赖统计学公式和较简单的代数运算（加减乘除）。

- **深度学习 (DL)**：

- - **自动特征学习 (Representation Learning)**：端到端（End-to-End）的学习。
  - *案例*：识别猫。直接把几万张猫的照片扔给DL模型，它自己学会了“哦，原来这种形状（耳朵）和那种纹理（毛发）组合起来就是猫”，不需要人教。
  - **多模型综合体**：一个神经网络本质上是无数个简单函数（神经元）的嵌套组合，形成一个超级复杂的函数。

#### 运算机制与规模

1. **矩阵运算 (Matrix Operations)**：

- - DL 的本质是**大规模矩阵乘法**（Matrix Multiplication）。
  - 输入是一个向量，权重是一个矩阵。![img](https://cdn.nlark.com/yuque/__latex/925e2427991ddde31c7fefd9d5168199.svg)。
  - 这就是为什么 DL 需要 **GPU**（图形处理器），因为 GPU 天生就是为了并行处理矩阵运算而设计的。

1. **数据规模 (Data Scale)**：

- - DL 是“数据饥渴型”技术。在小数据、高可解释性、低算力场景下，传统 ML（如 XGBoost、SVM）往往优于 DL。；但当数据量达到海量（Big Data）时，DL 的性能会由量变产生质变，远超传统 ML。

![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614573905-550f786c-e026-420f-8193-14847f0ea2fd.png)

------

### 4. 神经网络的工作原理（简单介绍）

神经网络的“学习”过程，其实就是**找参数（权重** ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg) **和偏置** ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)**）**的过程。

#### 两个关键阶段

1. **前向传播（推理阶段 / Guessing）**：

- - 数据输入 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 层层计算 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 得到预测结果。
  - *类比*：你蒙着眼睛投篮，凭感觉扔出去一个球（给出一个预测）。

1. **反向传播（学习阶段 / Learning）**：

- - **计算误差 (Loss)**：看预测结果和真实答案差多少（球偏离篮筐多远）。
  - **误差回传 (Backpropagation)**：将这个误差从输出层反向传回输入层。
  - **更新参数 (Optimizer)**：根据误差，告诉每一个神经元：“你刚才算错了，下次把权重调大/调小一点”。
  - *类比*：如果你投篮偏左了，大脑会告诉你“下次手臂往右偏一点用力”。经过成千上万次这样的“投篮-反馈-调整”，网络就“学会”了投篮。

## 神经网络的发展历程

人工智能（AI）和神经网络的发展并非一帆风顺，而是经历了“三起两落”。每一次低谷（AI Winter）通常都是因为当时的算法遇到了理论上的瓶颈或硬件算力不足。

### 1. 第一次浪潮：感知器时代 (1950s - 1960s)

**(The First Wave: The Perceptron Age)**

- **标志性事件**：1957年，Frank Rosenblatt 发明了**感知器（Perceptron）**。
- **当时的热潮**：

这是神经网络的“婴儿期”。当时人们非常乐观，认为只要把这些感知器连起来，计算机很快就能学会走路、说话、看东西。当时的感知器本质上是一个**单层**的神经网络。

#### 著名的 XOR 问题（异或问题）与第一次寒冬

- **转折点**：1969年，AI 领域的巨擘 Minsky 和 Papert 出版了一本叫《Perceptron》的书，直接给感知器泼了一盆冷水。
- **核心痛点**：他们从数学上证明了，**单层感知器甚至连最简单的 XOR（异或）逻辑都无法解决**。

- - **什么是 XOR？**

- - - AND（与）：两个都是1才输出1。
    - OR（或）：只要有一个是1就输出1。
    - **XOR（异或）**：两个**不一样**才输出1（例如 1和0 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 1；1和1 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 0）。

- **直观类比（线性不可分）**：

想象你在桌子上放了四颗棋子（代表 XOR 的四种情况）。

- - 单层感知器就像是一把直尺，它试图画**一条直线**把黑棋子和白棋子分开。
  - 对于 AND 和 OR 问题，你可以画一条线分开它们（这叫**线性可分**）。
  - 但对于 XOR 问题，无论你怎么摆弄这把直尺，都无法用**一条直线**分开它们。

- **结果**：因为连这么简单的逻辑都处理不了，政府和投资人停止了资助，AI 进入了长达十几年的**第一次寒冬**。

------

### 2. 第二次浪潮：反向传播复兴 (1980s - 1990s)

- **破局者**：Geoffrey Hinton 等人完善并推广了 **反向传播算法（Backpropagation, BP）**。
- **核心突破**：

1. 1. **多层感知器（MLP）**：既然一层网络画不出曲线，那就加一层！引入了**隐藏层（Hidden Layer）**。
   2. **非线性激活函数**：配合隐藏层，神经网络终于可以处理非线性问题了（解决了 XOR 问题）。
   3. **BP 算法**：解决了“多层网络如何训练”的难题，让误差可以一层层传回去修正参数。

#### 再次低谷：梯度消失与 SVM 的崛起

尽管解决了 XOR，但当时的神经网络并没有这就统治世界，反而又沉寂了。

- **技术瓶颈（梯度消失 Gradient Vanishing）**：

当时大家主要用 Sigmoid 激活函数。当网络层数变深时，误差反向传播回第一层时已经变得微乎其微（接近0），导致深层网络的前几层根本**学不到东西**。

- **外部竞争（SVM 的降维打击）**：

90年代中期，**支持向量机（SVM）** 和随机森林等算法表现非常出色。它们数学理论严谨（不像神经网络像个黑盒），计算量小，在当时的小规模数据集上效果完爆神经网络。

- **结果**：神经网络再次被打入冷宫，许多学者甚至不敢在论文标题里写“Neural Network”，生怕被拒稿。

------

### 3. 第三次浪潮：深度学习革命 (2006/2012 - Present)

这是我们目前正处的时代，AI 迎来了全面爆发。

- **爆发的三大引擎**：

1. 1. **大数据（Big Data）**：互联网普及，海量图片和文本数据为训练提供了“燃料”。
   2. **算力（Computation）**：**GPU（显卡）** 被发现非常适合跑神经网络（矩阵运算），计算速度提升了数十倍。
   3. **算法优化（Algorithms）**：解决了梯度消失问题（如使用 **ReLU** 激活函数），提出了更深的网络结构（如 ResNet, Transformer）。

- **里程碑事件**：

- - **2006年**：Hinton 提出“深度置信网络”，重新点燃希望（将 Neural Network 改名为 Deep Learning，听起来更高大上）。
  - **2012年 (AlexNet时刻)**：在 ImageNet 图像识别比赛中，深度学习算法以压倒性优势夺冠，从此深度学习一统江湖。

#### 生成对抗网络的简单介绍 (GAN) **(Generative Adversarial Networks)**

在 PPT 特别提到了 GAN，这是深度学习中非常有创意的模型。

- **核心思想：左右互搏** （左右脑互搏 dog）

GAN 由两个神经网络组成，它们像是一对宿敌，在博弈中共同进步。

- **通俗类比：造假币者 vs 警察**

1. 1. **生成器 (Generator, 造假者)**：

- - - 它的任务是凭空捏造数据（比如画一张假的人脸）。
    - **目标**：画得越像真的越好，目的是骗过判别器。

1. 1. **判别器 (Discriminator, 警察)**：

- - - 它的任务是接收图片，判断这张图是“真照片”还是“生成器画的假图”。
    - **目标**：火眼金睛，不被骗。

- **训练过程**：

- - 一开始，生成器画得很烂，判别器一眼识破。
  - 生成器挨骂后改进技术，画得逼真了一点；判别器为了不失业，也得提升鉴别能力。
  - **最终结果**：两者达到纳什均衡，生成器画出的图（如 AI 换脸、虚拟主播）连人类都无法分辨真假。

![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614602522-a1c27c91-bfac-4c6c-9a1a-3b731fd1a673.png)

------

## 神经网络的拓扑结构

**“拓扑结构”指的是神经元之间的连接模式**。不同的连接方式决定了信息流动的路径，也决定了网络擅长处理什么样的数据。

### 1. 网络结构的类型

#### (1) 前馈神经网络 (FNN, Feedforward Neural Networks)

这是最基础、最经典的架构，也是我们在入门阶段接触最多的类型。

- **结构特点**：

- - **单向流动**：信号从输入层 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 隐藏层 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 输出层，**永不回头**。
  - **无环路**：层与层之间全连接或局部连接，但没有反馈回路。
  - **无记忆**：处理当前输入时，与之前的输入毫无关系。

- **优缺点**：

- - ✅ **优点**：结构简单，易于理解和实现；训练相对稳定（反向传播路径清晰）。
  - ❌ **缺点**：无法处理序列信息（无法理解“上下文”）；通常要求输入数据的长度是固定的。

- **通俗类比**：

- - **自动贩卖机**：你投币，它吐出饮料。这一次购买和上一次购买完全没关系，机器没有“记忆”。
  - **做单选题**：做第一题和做第二题没有逻辑联系，不需要通过第一题的答案来推导第二题。

- **典型应用**：

- - **图像分类 (CNN)**：判断图片里是猫还是狗（CNN 虽然有卷积操作，但宏观上依然是前馈的）。
  - **回归预测**：根据房屋面积、位置预测房价。

#### (2) 循环神经网络 (RNN, Recurrent Neural Networks)

这是**NLP（自然语言处理）**的核心基石，专门为处理“顺序”而生。

- **结构特点**：

- - **循环回路**：隐藏层的输出不仅传给下一层，还**传回给自己**（作为下一时刻的输入）。
  - **有状态 (State)**：拥有“记忆”能力。当前的输出 = **当前的输入 + 上一时刻的记忆**。
  - **时间维度**：网络按时间步（Time Step）展开。

- **优缺点**：

- - ✅ **优点**：擅长处理变长的序列数据；能捕捉数据的前后依赖关系（上下文）。
  - ❌ **缺点**：**训练困难**（容易出现梯度消失或梯度爆炸）；**计算慢**（因为必须等上一时刻算完才能算下一时刻，难以并行化）。

- **通俗类比**：

- - **阅读理解**：你读到这一页时，对剧情的理解依赖于**前面几页**的内容。
  - **同声传译**：翻译后半句时，必须记得前半句说了什么。

- **典型应用**：

- - **自然语言处理**：机器翻译、文本生成、情感分析。
  - **语音识别**：将连续的语音波形转为文字。
  - **股票预测**：根据过去30天的走势预测明天的价格。

#### (3) 图神经网络 (GNN, Graph Neural Networks)

这是近年来处理**非欧几里得数据**（Non-Euclidean Data）的热门架构。

- **结构特点**：

- - **节点与边**：结构不再是分层的矩阵，而是由**节点 (Nodes)** 和 **边 (Edges)** 组成的网状结构。
  - **邻居聚合**：一个节点的状态更新，依赖于它所有**邻居节点**的信息聚合。

- **优缺点**：

- - ✅ **优点**：能处理复杂的非结构化数据（如社交关系、化学分子结构）。
  - ❌ **缺点**：计算复杂度高；大规模图数据的训练非常消耗资源。

- **通俗类比**：

- - **社交圈子**：**“你是谁，取决于你的朋友是谁”**。你的特征是你所有朋友特征的聚合（近朱者赤，近墨者黑）。
  - **病毒传播**：一个城市是否安全，取决于它连接的周边城市是否安全。

- **典型应用**：

- - **推荐系统**：基于用户和商品的交互图进行推荐。
  - **药物研发**：分析化学分子的原子结构（原子是节点，化学键是边）。
  - **知识图谱**：推理实体之间的关系。

#### (4) 混合架构 (Hybrid Architectures)

在实际的大型工业应用中，往往采用“组合拳”。

- **结构特点**：

- - **模块化组合**：将不同类型的网络串联或并联。例如：**Encoder-Decoder** 结构。
  - **各司其职**：用 CNN 提取空间特征，用 RNN 处理时间特征。

- **优缺点**：

- - ✅ **优点**：集各家之长，能解决单一模型无法解决的复杂多模态问题。
  - ❌ **缺点**：模型设计复杂，调参难度大（需要同时平衡两个网络的参数）。

- **通俗类比**：

- - **看图说话**：

- - - **眼睛 (CNN)**：负责看清楚图片里有什么（提取空间特征）。
    - **嘴巴 (RNN)**：负责把看到的东西组织成一句话说出来（生成序列文本）。

- **典型应用**：

- - **视频字幕生成 (Video Captioning)**：CNN 处理每一帧画面 + RNN 生成描述文字。
  - **视觉问答 (VQA)**：给一张图并问一个问题，机器回答。

#### 总结列表

| **架构类型**       | **核心关键词** | **记忆能力**  | **通俗类比**        | **典型应用领域**        |
| ------------------ | -------------- | ------------- | ------------------- | ----------------------- |
| **前馈网络 (FNN)** | 单向、无环     | **无** (静态) | 自动贩卖机 / 照相机 | 图像分类、简单回归      |
| **循环网络 (RNN)** | 循环、状态     | **有** (动态) | 阅读 / 录像机       | **NLP**、语音、股票预测 |
| **图网络 (GNN)**   | 节点、边、聚合 | 依赖邻居      | 朋友圈 / 社交关系   | 推荐系统、药物分子分析  |
| **混合架构**       | 组合、多模态   | 视组合而定    | 看图说话 (眼+嘴)    | 视频分析、图文生成      |

#### Python 代码示例

```python
import torch
import torch.nn as nn

# ==========================================
# 1. 前馈神经网络 (FNN/MLP) —— "直肠子"
# ==========================================
class FeedForwardNet(nn.Module):
    def __init__(self):
        # 初始化父类 nn.Module
        super(FeedForwardNet, self).__init__()

        # 定义第一层：全连接层
        # 输入维度 10 (特征数量)，输出维度 20 (隐藏层神经元数量)
        self.layer1 = nn.Linear(in_features=10, out_features=20) 

        # 定义第二层：输出层
        # 输入必须接上一层的输出 20，输出 2 (比如二分类任务)
        self.layer2 = nn.Linear(in_features=20, out_features=2)

    def forward(self, x):
        # x 的形状: [Batch_Size, 10]

        # 第一层计算 + ReLU 激活函数
        # 线性变换: [Batch, 10] -> [Batch, 20]
        # ReLU: 把负数变 0
        x = torch.relu(self.layer1(x))

        # 第二层计算 (输出层通常不加激活，或者在 Loss 中处理)
        # [Batch, 20] -> [Batch, 2]
        x = self.layer2(x)
        return x

# ==========================================
# 2. 循环神经网络 (RNN) —— "有记忆"
# ==========================================
class RecurrentNet(nn.Module):
    def __init__(self):
        super(RecurrentNet, self).__init__()

        # 定义 RNN 层
        # input_size=10: 每个时间步输入的特征数 (比如词向量维度)
        # hidden_size=20: 记忆单元的大小 (隐藏状态维度)
        # batch_first=True: 这是一个关键参数，表示输入数据的形状是 [Batch, Seq_Len, Feature]
        # 如果不加这个，默认输入需要是 [Seq_Len, Batch, Feature] (很容易搞混)
        self.rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

        # 定义全连接层：把 RNN 最后的记忆变成分类结果
        # 输入 20 (RNN的隐藏状态)，输出 2 (分类)
        self.fc = nn.Linear(in_features=20, out_features=2)

    def forward(self, x):
        # x 的形状: [Batch_Size, 8, 10] (假设序列长度为 8)

        # RNN 前向传播
        # out: 包含了每一个时间步的输出，形状 [Batch, 8, 20]
        # h_n: 只包含最后一个时间步的隐藏状态 (记忆)，形状 [1, Batch, 20]
        out, h_n = self.rnn(x) 

        # 我们做分类通常只需要看完最后一步后的"读后感"
        # 取出 out 的最后一个时间步的数据 (索引 -1)
        # out[:, -1, :] 的形状变成 [Batch, 20]
        last_step_output = out[:, -1, :]

        # 送入全连接层分类
        return self.fc(last_step_output) 

# ==========================================
# 3. 卷积神经网络 (CNN) —— "局部感知"
# ==========================================
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 定义卷积层
        # in_channels=3: 输入图片的通道数 (RGB彩色图)
        # out_channels=16: 我们想提取 16 种不同的特征 (对应 16 个卷积核)
        # kernel_size=3: 卷积核大小 3x3
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

        # 定义池化层
        # 2x2 的最大池化，把图片长宽砍半
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 定义全连接层 (分类器)
        # 这里的输入维度需要手动算一下：
        # 1. 输入图片假设 32x32
        # 2. 经过 Conv (无padding, 3x3核): (32-3)/1 + 1 = 30x30
        # 3. 经过 Pool (2x2): 30/2 = 15x15
        # 4. 也就是最后变成 16个通道 x 15 x 15
        self.fc = nn.Linear(in_features=16 * 15 * 15, out_features=2)

    def forward(self, x):
        # x 的形状: [Batch, 3, 32, 32]

        # 卷积 -> ReLU -> 池化
        # Conv: [Batch, 16, 30, 30]
        # ReLU: 数值变非线性
        # Pool: [Batch, 16, 15, 15]
        x = self.pool(torch.relu(self.conv(x)))
        
        # 展平 (Flatten): 把立体的特征图拉成一维向量
        # view(Batch, -1) 表示自动计算后面的维度
        # 形状变为: [Batch, 16*15*15] = [Batch, 3600]
        x = x.view(x.size(0), -1) 
        
        # 全连接分类
        x = self.fc(x)
        return x

# ==========================================
# 模拟输入数据 (观察维度的不同！)
# ==========================================

# 1. FNN 数据: 二维 [样本数, 特征数]
input_fnn = torch.randn(5, 10)
print(f"FNN 输入形状: {input_fnn.shape}")

# 2. RNN 数据: 三维 [样本数, 序列长度(时间步), 特征数]
# 注意：nn.RNN 默认 不支持 batch_first=True！这是 nn.LSTM 和 nn.GRU 才有的参数。
# 		nn.RNN 的输入必须是 [seq_len, batch, input_size]，否则会报错。
# 比如: 5句话，每句话8个词，每个词用长度10的向量表示
input_rnn = torch.randn(5, 8, 10) 
print(f"RNN 输入形状: {input_rnn.shape}")

# 3. CNN 数据: 四维 [样本数, 通道数, 高, 宽]
# 比如: 5张图，RGB 3通道，分辨率 32x32
input_cnn = torch.randn(5, 3, 32, 32)
print(f"CNN 输入形状: {input_cnn.shape}")
```

1. **FNN (全连接)**：

- - 处理 **表格/向量** 数据。
  - 输入形状是 **2D**：`(Batch, Features)`。

1. **RNN (循环)**：

- - 处理 **文本/语音/时间序列** 数据。
  - 输入形状是 **3D**：`(Batch, Sequence, Features)`。
  - *注意*：`batch_first=True` 很重要，否则第一维是序列长度。

1. **CNN (卷积)**：

- - 处理 **图像/视频** 数据。
  - 输入形状是 **4D**：`(Batch, Channel, Height, Width)`。
  - *注意*：进入全连接层前，必须用 `.view()` 把数据拍扁（展平）。

------

### 时序数据与时序问题的简单介绍

我们在课堂中提到了 NLP 的特征提取。这里我们需要从数据的角度理解：**为什么 NLP 这么难？** 因为文本是**时序数据**。

#### 1. 什么是时序数据？

凡是**前后顺序不可打乱**，且**前后存在因果/依赖关系**的数据，都是时序数据。

- **自然语言**：汉字单独拿出来没有意义，组合在一起且顺序正确才有意义。
- **语音音频**：一段波形，切碎了就是噪音。
- **股票/天气**：今天的温度受昨天影响。

#### 2. 为什么需要特殊的网络（RNN）来处理？

如果使用普通的前馈网络（如词袋模型 Bag of Words），会丢失“顺序”这个关键信息。

- **案例对比**：

- - 句子 A：**“屡战屡败”** (Fight repeatedly, lose repeatedly - 这是一个悲伤的故事)
  - 句子 B：**“屡败屡战”** (Lose repeatedly, fight repeatedly - 这是一个励志的故事)

- **网络的视角**：

- - **普通网络 (FNN)**：看到四个字 {屡, 战, 屡, 败}。它认为这两个句子是一模一样的（特征向量相同）。
  - **时序网络 (RNN)**：它按顺序读。

- - - 读句子 A：先记录“败”的状态，最后结束在“败”上 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 判定为消极。
    - 读句子 B：虽然也有“败”，但最后结束在“战”上 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 判定为积极。

**总结**：在 NLP 中，**顺序 (Sequence)** 和 **上下文 (Context)** 就是一切。这就是为什么我们在后面要重点学习 RNN、LSTM 以及 Transformer 的原因。

#### ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614627326-62bec483-4ce1-464b-9352-9bc888918357.png)

------

## 感知器和激活函数

### 1. 感知器详解 (The Perceptron) —— 神经网络的“原子”

#### (1) 什么是感知器？

**感知器（Perceptron）** 是最简单、最古老的人工神经网络模型，由 Frank Rosenblatt 在 1957 年提出。

你可以把它看作是**只有一个神经元**的神经网络。它是现代深度学习所有复杂架构的基石。

#### (2) 工作原理：从生物到数学

我们在“基本概念”一节中做过类比，这里我们从数学角度精确拆解它的**五个核心组件**：

1. **输入 (Inputs,** ![img](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg)**)**：外界传入的信息（比如图片的像素值，或者上一层的输出）。
2. **权重 (Weights,** ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg)**)**：代表输入信号的**重要程度**。

- - 比如决定“是否买房”，![img](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg)是价格，![img](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg)是楼层。如果不差钱但怕高，那么 ![img](https://cdn.nlark.com/yuque/__latex/204ef929f76b3378e50aa9551cb332a3.svg)（价格权重）就低，![img](https://cdn.nlark.com/yuque/__latex/84d86c0da655fb4aa82d8ca0151c603b.svg)（楼层权重）就高。

1. **加权求和 (Weighted Sum,** ![img](https://cdn.nlark.com/yuque/__latex/6bec5bf43c9924bf5006e907b643f00c.svg)**)**：

![img](https://cdn.nlark.com/yuque/__latex/bb2e5463627571c9aa2a9f5faf210342.svg)

这是神经元对输入信息进行“汇总”的过程。

1. **偏置 (Bias,** ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)**)**：

- - **作用**：调整激活的**阈值**。
  - **通俗理解**：它是你内心的“预设立场”。如果 ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg) 很大（正偏置），说明你很容易被说服（容易输出 1）；如果 ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg) 很小（负偏置），说明你很固执，需要非常强的输入信号才能打动你。

1. **阶跃函数 (Step Function)**：

- - 最早期的感知器并没有现在这么复杂的激活函数（如 ReLU），它用的是最简单的**阶跃函数**：
  - **规则**：如果加权和 ![img](https://cdn.nlark.com/yuque/__latex/663659c43dff452f8abd1ddce4232a11.svg)，输出 1；否则输出 0。

#### (3) 通俗案例：要不要去吃火锅？

感知器其实就是一个**线性分类器**，它在做简单的决策。

- **场景**：你要决定今晚是否去吃火锅（Output: 1去, 0不去）。
- **输入特征**：

- - ![img](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg)：有朋友陪吗？ (1有, 0无)
  - ![img](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg)：排队人多吗？ (1多, 0少)

- **你的权重（偏好）**：

- - ![img](https://cdn.nlark.com/yuque/__latex/f166c946ff9ae0bf07d8902a9a76b312.svg)：你很看重朋友陪伴。
  - ![img](https://cdn.nlark.com/yuque/__latex/5e98f10b0c19d6b83ba64f9ea98ec4be.svg)：你讨厌排队（负分）。
  - ![img](https://cdn.nlark.com/yuque/__latex/ba15188220940e46e7a44a35efdc5b15.svg)：你的基础门槛（比如你要减肥，起步就是负分）。

- **计算过程**：

![img](https://cdn.nlark.com/yuque/__latex/5d4775c63e0fe785fb2b4693805cc4a5.svg)

- - **情况A（有朋友，排队多）**：![img](https://cdn.nlark.com/yuque/__latex/342e7cce4fd9175d74e4df507f67e68e.svg) ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) **去！**
  - **情况B（没朋友，排队少）**：![img](https://cdn.nlark.com/yuque/__latex/805857779129eade9021985864276e72.svg) ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) **不去。**

#### (4) 感知器的致命缺陷：XOR 问题 (异或问题)

这是 AI 历史上著名的“滑铁卢”，直接导致了第一次 AI 寒冬。

- **能力边界**：单层感知器只能解决**线性可分 (Linearly Separable)** 的问题。

- - 也就是能用**一条直线**（或平面）把两类数据完全分开。
  - 比如 AND（与）和 OR（或）逻辑，画一条线就能分开 0 和 1。

- **XOR 困境**：

- - **异或逻辑**：当两个输入**不同**时为 1（真），**相同**时为 0（假）。
  - ![img](https://cdn.nlark.com/yuque/__latex/c32fd6d5af9892ee4eb8ee50933527ac.svg)
  - ![img](https://cdn.nlark.com/yuque/__latex/71826643d1e1820a6cf05887bd59c63c.svg)
  - ![img](https://cdn.nlark.com/yuque/__latex/e52ee88b0d64b984b492c0efad1564a2.svg)
  - ![img](https://cdn.nlark.com/yuque/__latex/f2cb211942b32354f68153cf433116f2.svg)
  - **图解**：想象在一张纸上画这四个点，你绝对无法只画**一条直线**就把两个 1 和两个 0 分开。

- **结论**：单层感知器连简单的异或逻辑都学不会。这也是后来为什么要引入**隐藏层**和**非线性激活函数**的原因。

------

### 2. Python 代码示例：一个简单的感知器

这段代码演示了感知器如何“学会”简单的逻辑（如 AND 门），以及它为什么学不会 XOR。

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.W = np.zeros(input_size + 1) # +1 是因为包含偏置 bias
        self.lr = lr       # 学习率
        self.epochs = epochs

    def activation_fn(self, x):
        # 阶跃函数：大于0输出1，否则输出0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # 插入 1 作为偏置项的输入
        z = self.W.T.dot(np.insert(x, 0, 1))
        return self.activation_fn(z)

    def train(self, X, d):
        # 简单的感知器学习规则
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.activation_fn(self.W.T.dot(x))
                e = d[i] - y # 计算误差
                self.W = self.W + self.lr * e * x # 更新权重

# ------------------------------------
# 1. 训练它学习 AND 逻辑 (线性可分，能学会)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
d_and = np.array([0, 0, 0, 1]) # 只有 1,1 才是 1

perceptron = Perceptron(input_size=2)
perceptron.train(X, d_and)
print("AND 逻辑预测结果:", [perceptron.predict(x) for x in X]) 
# 输出应该是 [0, 0, 0, 1] -> 成功！

# ------------------------------------
# 2. 训练它学习 XOR 逻辑 (线性不可分，学不会)
d_xor = np.array([0, 1, 1, 0]) # 相同为0，不同为1

perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X, d_xor)
print("XOR 逻辑预测结果:", [perceptron_xor.predict(x) for x in X])
# 输出往往是错误的，比如 [0, 0, 0, 0] 或 [1, 1, 1, 0]，永远无法全对。
```

### 2. 激活函数详解 (Activation Functions) —— 神经网络的灵魂

#### 核心概念：它是什么？为什么要用它？

- **定义**： 激活函数（Activation Function）是添加在神经元输出端的一个**非线性函数**。 如果把神经元比作一个水龙头，输入信号是水流，权重是水管的粗细，那么激活函数就是**龙头的开关**。它根据流入的水量（加权和）决定最后流出多少水，甚至决定要不要流出水（是否被激活）。
- **为什么要引入非线性？（面试/考试必问）** 如果在神经网络中不使用激活函数，或者使用线性的激活函数（比如 ![img](https://cdn.nlark.com/yuque/__latex/78f60782ca614e86d5a24efce34c149d.svg)），那么无论你的网络有多少层（10层、100层），它本质上都等价于**一个单层的线性模型**。

- - **数学推导**： 假设第一层输出 ![img](https://cdn.nlark.com/yuque/__latex/496f57501300900d44eee60d69954add.svg)，第二层输出 ![img](https://cdn.nlark.com/yuque/__latex/eda42170114c88bcf6349fdeb61f1c7b.svg)。 代入后：![img](https://cdn.nlark.com/yuque/__latex/00b7b3f37b6c90b533a51119818471e0.svg)。 你看，两层矩阵相乘，最后还是变成了一个矩阵。线性函数的叠加依然是线性的。
  - **通俗类比**：

- - - **没有激活函数（线性）**：就像你手里只有一把**直尺**。无论你怎么拼接直尺，你画出来的永远是由直线组成的图形，画不出圆，也画不出复杂的波浪线（非线性分类边界）。
    - **有了激活函数（非线性）**：就像把直尺**折弯**了，或者允许你徒手画线。这样神经网络才能逼近任意复杂的函数（Universal Approximation Theorem，万能逼近定理）。

#### 挑选激活函数的标准

并不是随便找个函数就能当激活函数的，它通常需要满足以下条件：

1. **非线性 (Non-linear)**：这是必须的，为了拟合复杂曲线。
2. **可导性 (Differentiable)**：在反向传播时，我们需要算梯度（求导），所以函数最好处处可导（或者几乎处处可导，如ReLU）。

**两个著名的“拦路虎”问题**： 在选择激活函数时，我们最怕遇到这两个问题：

- **梯度消失 (Gradient Vanishing)**：

- - *现象*：导数非常小（趋近于0）。
  - *后果*：在深层网络中，梯度是从后往前传的（链式法则连乘）。![img](https://cdn.nlark.com/yuque/__latex/7896a72e77b867ef0eadd348fd1446c7.svg)。传到最后，前面层的参数更新量几乎为0，网络**“学不动”**了。

- **梯度爆炸 (Gradient Exploding)**：

- - *现象*：导数非常大（大于1）。
  - *后果*：![img](https://cdn.nlark.com/yuque/__latex/bab637137b08713a42c015ec2c04256f.svg)。参数更新幅度过大，网络数值溢出（NaN），直接崩溃。

------

#### 常见的激活函数详解

我们将从**原理、公式、导数（梯度）、优缺点**以及**代码实现**五个维度，对 Sigmoid, Tanh, ReLU, Softmax 进行深度拆解。

##### (1) Sigmoid 函数 —— “曾经的贵族，如今的配角”

这是神经网络早期最常用的函数，模拟了生物神经元的“全有或全无”特性（但这其实是个误解，生物神经元比这复杂得多）。

- **数学公式**：

![img](https://cdn.nlark.com/yuque/__latex/9e1142b0bd944b0807e3c54ea79bbe4c.svg)

- **导数公式（敲黑板，反向传播用）**：

![img](https://cdn.nlark.com/yuque/__latex/3b7749ecd1ac8789ee507e49392eed73.svg)

- - *注意*：导数最大值只有 **0.25**（当 ![img](https://cdn.nlark.com/yuque/__latex/25e542ef63bbd4f97b8db971163c2d18.svg) 时）。这意味着每传一层，梯度至少缩水 75%。

- **图像特征**：

- - ![img](https://cdn.nlark.com/yuque/0/2026/jpeg/35927690/1769612709809-3d42fee1-a7d2-4434-8d8b-1196de65fb29.jpeg)
  - 输入 ![img](https://cdn.nlark.com/yuque/__latex/222514078f9382e056882aa57af44111.svg) 输出 1；输入 ![img](https://cdn.nlark.com/yuque/__latex/cd36c215be932293b8b8ff50b6afa5d1.svg) 输出 0。

- **深度剖析：为什么现在很少用它？**

1. 1. **梯度消失 (Gradient Vanishing) —— 致命伤**：

- - - 当输入 ![img](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg) 很大（比如 10）或很小（比如 -10）时，Sigmoid 曲线几乎是平的。这意味着**导数趋近于 0**。
    - 在深层网络中，梯度是连乘的。如果每一层的梯度都小于 0.25，甚至接近 0，传了几层之后，梯度就消失了。**底层的参数得不到更新，网络就学不到任何特征**。

1. 1. **非零中心 (Non-Zero Centered)**：

- - - Sigmoid 的输出永远是正数 ![img](https://cdn.nlark.com/yuque/__latex/60595188a4cc7b200de5a6f8243a2127.svg)。
    - 这会导致下一层神经元的输入全是正数，进而导致权重 ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg) 的梯度更新方向要么全是正，要么全是负（Z字形下降），导致收敛速度变慢。

1. 1. **计算昂贵**：

- - - `exp()` 指数运算在计算机底层是泰勒展开级数求和，比简单的加减乘除慢得多。

- **最佳适用场景**：

- - **二分类问题的输出层**（因为它直接输出概率）。
  - **几乎不再用于隐藏层**。

##### (2) Tanh 函数 (Hyperbolic Tangent) —— “Sigmoid 的改进版”

为了解决 Sigmoid“非零中心”的问题，数学家搬出了双曲正切函数。

- **数学公式**：

![img](https://cdn.nlark.com/yuque/__latex/a8012aa75af8147d3159b3c2a8ec955c.svg)

- **导数公式**：

![img](https://cdn.nlark.com/yuque/__latex/b87e2e8a2865ff45e57e5b77d72ebb66.svg)

- - *注意*：导数最大值是 **1**（当 ![img](https://cdn.nlark.com/yuque/__latex/25e542ef63bbd4f97b8db971163c2d18.svg) 时）。这一点比 Sigmoid 好很多。

- **图像特征**：

- - ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769612762118-e4b09366-170c-4202-b99c-c7a03510434c.png)
  - 形状和 Sigmoid 很像，但是范围拉伸到了 **[-1, 1]**。
  - 穿过原点 ![img](https://cdn.nlark.com/yuque/__latex/79e4b67771e7db6b16990e64a36c6521.svg)。

- **深度剖析：优缺点**

- - **✅** **优点**：它是 **Zero-Centered（以0为中心）** 的。正负输入能产生正负输出，梯度的更新方向更健康，收敛比 Sigmoid 快。
  - **❌** **缺点**：**梯度消失问题依然存在**。看图就知道，当 ![img](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg) 很大或很小时，它依然是平的（饱和区），导数依然是 0。

- **最佳适用场景**：

- - **循环神经网络 (RNN, LSTM, GRU)**。在这些处理序列的模型中，Tanh 依然是默认配置，因为它能把内部状态限制在 [-1, 1] 之间，防止数值无限膨胀。

##### (3) ReLU 函数 (Rectified Linear Unit) —— “简单粗暴的现代之王”

这是目前深度学习（包括 CNN, Transformer, 大模型）中**最核心、最常用**的激活函数。它的逻辑非常简单：**“是正数就通过，是负数就置0”**。

- **数学公式**：

![img](https://cdn.nlark.com/yuque/__latex/a14f377ba0a162f83ed2387631acfba7.svg)

- **导数公式**：

![img](https://cdn.nlark.com/yuque/__latex/92f37ce8877995646e473b3d532c6bee.svg)

- **图像特征**：

- - ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769612906712-5d6a369d-0ff1-4658-9235-f617e8a7853c.png)
  - 左边是死水一潭（0），右边是一路高歌（线性）。

- **深度剖析：为什么它能统治深度学习？**

1. 1. **彻底解决正区间的梯度消失**：

- - - 只要输入是正数，导数永远是 **1**。
    - 无论网络有多深，梯度 ![img](https://cdn.nlark.com/yuque/__latex/cb06d79cec1dd24ad1c8435594ec08a9.svg)，信号可以原封不动地传回第一层。这是深层网络（Deep Learning）能训练起来的关键。

1. 1. **计算极快**：

- - - 没有指数，没有除法。计算机只需要做一个判断：`if x > 0`。在大规模矩阵运算中，这能节省海量时间。

1. 1. **稀疏性 (Sparsity)**：

- - - 在很多时候，负数输入会让神经元输出 0（不激活）。这意味着网络中只有一部分神经元在工作，这模拟了生物大脑的节能和稀疏特性，能减少过拟合。

- **潜在风险：Dead ReLU Problem (神经元死亡)**

- - **现象**：如果学习率太大，或者运气不好，某个神经元的权重更新后，使得对于所有的输入数据，加权和都是负数。
  - **结果**：该神经元输出永远是 0，导数永远是 0。**这个神经元从此“死”了，再也不会更新参数**。
  - **对策**：使用 Leaky ReLU（给负区一点点斜率，比如 0.01）或者 GeLU。

- **最佳适用场景**：

- - **几乎所有深度网络的隐藏层**（CNN, ResNet, Transformer 等）。

##### (4) Softmax 函数 —— “多路归一化”

它和前面三个不一样。前面三个是**单打独斗**（只看当前神经元的输入），Softmax 是**统筹全局**（看这一层所有神经元的关系）。

- **数学公式**：

![img](https://cdn.nlark.com/yuque/__latex/3a6b1f352e638b9d80dea6c853981f5a.svg)

- - ![img](https://cdn.nlark.com/yuque/__latex/588455b04479646f6b7a4a2886b01dba.svg) 是当前类别的得分，![img](https://cdn.nlark.com/yuque/__latex/ba1b6e7a2a5f6e2808e06f1b6133bb82.svg) 是当前类别的概率。

- **通俗理解**：

它做了三件事：

1. 1. **指数化 (**![img](https://cdn.nlark.com/yuque/__latex/49cc400b77e6d24de7cc156670e4371c.svg)**)**：把负数分值变成正数，同时拉大差距（分值高一点点，概率大很多）。
   2. **归一化 (**![img](https://cdn.nlark.com/yuque/__latex/6bec5bf43c9924bf5006e907b643f00c.svg)**)**：让所有类别的概率加起来等于 1。
   3. **概率化**：把生硬的分数（Logits）变成“这是猫的概率是 80%”。

- **深度剖析：为什么叫 "Soft" max?**

- - **Hard Max**：[2, 5, 1] ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) [0, 1, 0]。只选最大的，其他全杀掉。太绝对，不可导。
  - **Soft Max**：[2, 5, 1] ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) [0.04, 0.95, 0.01]。最大的概率最大，但其他人也有机会。它是平滑的、可导的。

- **最佳适用场景**：

- - **多分类问题的输出层**（ImageNet 1000分类，BERT 的词表预测）。

##### ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614660299-932c883d-befa-4b51-9029-3ad3ddce56c9.png)

------

#### Python 代码示例

这是一段核心代码，展示了它们在 PyTorch 中是如何被调用的，以及处理数据的差异。

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 模拟数据：从 -10 到 10 的 100 个数字
x = torch.linspace(-5, 5, 200)

# 1. Sigmoid：把数值压缩到 0~1，负数区梯度很小（消失风险）
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)

# 2. Tanh：负数变0（稀疏性），正数不变（梯度畅通）
tanh = nn.Tanh()
y_tanh = tanh(x)

# 3. ReLU
relu = nn.ReLU()
y_relu = relu(x)

# 4. Softmax 多分类概率归一化 (通常用于输出层)(注意：通常用于多维数据，这里模拟一个批次)
# 假设有3个类别，分数分别是 [2.0, 1.0, 0.1]
logits = torch.tensor([[2.0, 1.0, 0.1]])
softmax = nn.Softmax(dim=1)
probs = softmax(logits)

print(f"Softmax 输入分数: {logits}")
print(f"Softmax 输出概率: {probs}")
# 输出类似于: tensor([[0.6590, 0.2420, 0.0990]])，加起来是1

# 绘图代码（可直接运行查看形状）
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid (0~1)', linestyle='--')
plt.plot(x.numpy(), y_tanh.numpy(), label='Tanh (-1~1)')
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU (0~inf)', linewidth=2)
plt.title('Activation Functions Comparison')
plt.grid(True)
plt.legend()
plt.show()
```

#### 总结 (Summary Cheat Sheet)

| **激活函数** | **核心特点** | **优点**                 | **缺点**             | **最佳适用位置**          |
| ------------ | ------------ | ------------------------ | -------------------- | ------------------------- |
| **Sigmoid**  | S形，(0,1)   | 概率解释                 | **梯度消失**，计算慢 | 二分类输出层              |
| **Tanh**     | S形，(-1,1)  | 零中心 (Zero-centered)   | 梯度消失             | RNN/LSTM 隐藏层           |
| **ReLU**     | 折线，(0,+∞) | **无梯度消失**，计算极快 | Dead ReLU (死神经元) | CNN/LLM 隐藏层 (默认首选) |
| **Softmax**  | 归一化       | 输出概率和为1            | 计算复杂             | 多分类输出层              |

## 数据的传播机制

神经网络的工作流程本质上就是**双向**的数据流动：一个是**前向传播**（去程），一个是**反向传播**（回程）。

### 1. 前向传播 (Forward Propagation) —— “数据的推理过程”

#### (1) 定义

前向传播是指数据从**输入层**开始，经过一层层**隐藏层**的计算（加权求和 + 激活），最后到达**输出层**得到预测结果的过程。

- **流向**：`输入层 (Input)` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `隐藏层 (Hidden)` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `输出层 (Output)`
- **核心动作**：**计算 (Calculation)**。
- **本质**：将原始数据映射为结果。

![img](https://cdn.nlark.com/yuque/__latex/c7c7798bf0a0e31ff177f07a841f875b.svg)

#### (2) 通俗类比：工厂流水线

- **原材料进厂（输入层）**：比如面粉、糖、水。
- **各车间加工（隐藏层）**：

- - 车间1：混合搅拌（加权求和）。
  - 车间2：烘烤成型（激活函数）。

- **成品出厂（输出层）**：生产出一块饼干（预测结果）。
- **特点**：在这个过程中，数据一旦流过某个车间，就变成新的形态，**不会倒流**。

------

### 2. 反向传播 (Backward Propagation) —— “误差的归因过程”

这是深度学习中最核心、最精妙的部分。它的目的是为了解决**“责任分配”**的问题。

#### (1) 定义

反向传播是指将输出层产生的**误差 (Error)**，按照网络的连接路径，**由后向前**一层层传回输入层的过程。

- **流向**：`输出层 (Output)` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `隐藏层 (Hidden)` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `输入层 (Input)`
- **核心动作**：**求导 (Differentiation)**。
- **核心工具**：**链式法则 (Chain Rule)**。

#### (2) 为什么需要反向传播？

当饼干做出来（前向传播结束）后，我们发现饼干**太甜了**（产生了误差）。

现在的任务是：**我们要找出是谁的责任，以便下次改进。**

- 是烘烤的时间太长了？
- 还是糖放多了？
- 还是面粉买错了？

反向传播就是从成品开始，倒着查流水线：

1. **先查最后一步**：成品太甜 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 可能是糖浆涂多了。
2. **再查中间一步**：如果糖浆没问题 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 可能是面团里的糖放多了。
3. **最后查源头**：如果配方没问题 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 可能是采购的糖本身太甜。

#### (3) 核心产物：梯度 (Gradient)

反向传播计算出来的东西叫做**“梯度”**。

- **梯度的含义**：它告诉每一个神经元的权重参数 ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg)：“你刚才对误差贡献了多少”。

- - 如果梯度很大，说明你这个参数是造成错误的主要原因，下次要狠狠地改。
  - 如果梯度很小，说明这个错误跟你没啥关系，下次保持原样就行。

------

### 3. 总结与对比 (Summary)

| **特性**     | **前向传播 (Forward)**                                       | **反向传播 (Backward)**                                      |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **方向**     | 从头到尾 (![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg)) | 从尾到头 (![img](https://cdn.nlark.com/yuque/__latex/1c4b6f6d50a08c763be1abeca063a01f.svg)) |
| **输入**     | 原始数据 (Data)                                              | 误差信号 (Error)                                             |
| **输出**     | 预测结果 (Prediction)                                        | 参数梯度 (Gradients)                                         |
| **目的**     | **“猜”** (Inference)                                         | **“学”** (Learning 的基础)                                   |
| **数学操作** | 乘法、加法、激活函数                                         | 偏导数、链式法则                                             |

------

## 文本预处理流程

整个预处理流程就像是**烹饪前的食材处理**：你不能把整头牛直接扔进锅里，你需要清洗、去皮、切块，最后才能变成这道菜的原材料。

### 1. 数据清洗 (Data Cleaning) —— “洗菜”

这是第一步，目的是去除所有的“噪音”。

- **做什么**：

- - **去除 HTML 标签**：如果你爬取的是网页数据，会有很多 `<div>`, `<br>` 等无意义的符号。
  - **去除特殊符号**：如表情包、乱码、非文本字符（除非你在做情感分析，表情包可能有用）。
  - **大小写转换 (Lowercasing)**：在英文中，"Apple" 和 "apple" 应该被视为同一个词，通常统一转为小写。

### 2. 分词 (Tokenization) —— “切菜”

这是将长文本拆解成最小单位的过程。

- **定义**：将句子切分成一个个独立的**词 (Token)**。
- **中英文差异**：

- - **英文**：很简单，天然有空格分隔。`"I love AI"` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `['I', 'love', 'AI']`。
  - **中文**：很难，词与词之间没有空格。需要使用分词工具（如 **Jieba** 分词）。
  - *案例*：“下雨天留客天留我不留”。如果不分词，机器完全看不懂；分词后：“下雨天 / 留客 / 天 / 留 / 我 / 不 / 留”。

### 3. 停用词过滤 (Stop Words Removal) —— “去杂质”

有些词虽然出现频率极高，但对于理解句意没有帮助，甚至会干扰模型。

- **什么是停用词**：

- - **英文**：the, is, at, which, on...
  - **中文**：的、了、么、呢、啊...

- **处理方式**：直接建立一个“停用词表”，把分词结果中属于这个表里的词统统删掉。
- *类比*：从沙子里淘金，我们要的是金子（关键词），沙子（停用词）要筛掉。

### 4. 词干提取与词形还原 (Stemming & Lemmatization) —— “标准化”

这一步主要针对英文等多态语言。目的是把词的不同形态统一成一个标准形态。

- **词干提取 (Stemming)**：

- - **做法**：简单粗暴地**切掉**词缀。
  - *例子*：`running`, `runs`, `ran` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `run`。
  - *缺点*：可能切出不存在的词（比如 `universities` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `univers`）。

- **词形还原 (Lemmatization)**：

- - **做法**：更高级，基于词典，把词还原为**原形**。
  - *例子*：`better` ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) `good`（Stemming 做不到这一点）。

- *注*：中文一般不需要这一步，因为汉字没有时态变化。

------

### 5. 特征提取 / 向量化 (Feature Extraction / Vectorization)

**这一步非常重要！**经过上面四步，我们得到了一堆干净的词。现在要把它们变成**数字**。

#### (1) 词袋模型 (BoW, Bag of Words)

这是最简单、最直观的方法。

- **原理**：不考虑词的顺序，只统计**词出现的次数**。
- **过程**：

1. 1. 建立一个包含所有词的**词典**（Vocabulary）。
   2. 数一数每个句子里，词典里的词出现了几次。

- **案例**：

- - 句子 A：`I love AI`
  - 句子 B：`AI is future`
  - 词典：`[I, love, AI, is, future]` (共5个词)
  - **向量 A**：`[1, 1, 1, 0, 0]`
  - **向量 B**：`[0, 0, 1, 1, 1]`

- **缺点**：生成的向量非常稀疏（大部分是0），且丢失了语序信息。

#### (2) TF-IDF (词频-逆文档频率)

BoW 有个大问题：有些词（比如“总是”、“非常”）出现的次数很多，但其实没啥重要信息。TF-IDF 就是为了解决这个问题：**给重要的词高权重，给不重要的词低权重。**

- **公式核心思想**：

- - **TF (Term Frequency)**：这个词在这篇文章里出现的越多，越重要。
  - **IDF (Inverse Document Frequency)**：这个词在**所有**文章里出现的越少，越重要（说明它是独有的、能代表本文特征的）。
  - **最终权重** = ![img](https://cdn.nlark.com/yuque/__latex/e3d7b233d1804f0d7ed8c0e8b7979ce9.svg)。

- **通俗类比**：

- - 词语 **"神经网络"**：在本节课笔记里频繁出现（TF高），在菜谱大全里几乎不出现（IDF高） ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) **权重极高，是关键词**。
  - 词语 **"我们"**：在本节课笔记里频繁出现（TF高），但在所有中文文章里都频繁出现（IDF极低，接近0） ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) **权重低，不是关键词**。

### ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614722845-76ba4de7-673d-49a4-ae53-0e809ff983fb.png)

------

### Python 代码示例

```python
import jieba

def clean_text(text):
    # 1. 假设这是我们的停用词表 (实际项目中会从文件读取)
    stop_words = {"的", "是", "了", "在", "和"}
    
    # 2. 分词
    words = jieba.lcut(text)
    
    # 3. 过滤 (列表推导式)
    # 只保留: 不在停用词表中 且 长度大于1 的词
    result = [w for w in words if w not in stop_words and len(w) > 1]
    
    return result

# 测试
raw_text = "深度学习是人工智能的一个重要分支，我在学习它。"
print(f"原始: {raw_text}")
print(f"处理后: {clean_text(raw_text)}")
# 输出: ['深度', '学习', '人工智能', '一个', '重要', '分支', '学习']
# 这里的 "是", "的", "在", "它"(长度1) 都被过滤了
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. 定义语料库 (Corpus)
# 故意造句：让 "AI" 重复出现，让 "is" 这种无意义词也出现
corpus = [
    "AI is great.",             # 句子 1
    "I love AI and AI loves me.", # 句子 2 (AI 出现两次)
    "The future is AI."         # 句子 3
]

# -------------------------------------------------------
# 2. 选手一：词袋模型 (Bag of Words) - 只看数量
# -------------------------------------------------------
bow_vec = CountVectorizer()
bow_matrix = bow_vec.fit_transform(corpus)

# 转成 DataFrame 方便查看
df_bow = pd.DataFrame(bow_matrix.toarray(), 
                      columns=bow_vec.get_feature_names_out(),
                      index=["句1", "句2", "句3"])

print(">>> 词袋模型 (BoW) - 统计词频：")
print(df_bow)
# 观察点：在“句2”中，'ai' 的值是 2。它认为出现次数越多越重要。

# -------------------------------------------------------
# 3. 选手二：TF-IDF - 看重“独特性”
# -------------------------------------------------------
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)

df_tfidf = pd.DataFrame(tfidf_matrix.toarray().round(2), 
                        columns=tfidf_vec.get_feature_names_out(),
                        index=["句1", "句2", "句3"])

print("\n>>> TF-IDF - 统计权重：")
print(df_tfidf)
# 观察点：
# 1. 虽然 'is' 在句1和句3都出现了，但它的权重(约0.4-0.5)通常低于该句中的核心词(如 great)。
# 2. 'ai' 虽然出现频率高，但因为在每个句子里都有(不稀缺)，它的 IDF 值会被拉低，
#    导致最终权重可能不如只在句2出现的 'love' 高。
```

------

## 梯度下降算法

### 1. 数学中的最小值探索

在纯数学的世界里，寻找函数的最小值通常有两种路径：**解析解**和**数值解**。

#### (1) 解析法 (Analytical Solution) —— “上帝视角”

这是最直接、最精确的方法。

- **原理**：利用微积分中的**费马引理**。函数极值点的导数（斜率）一定为 0。
- **步骤**：

1. 1. 对目标函数求导：![img](https://cdn.nlark.com/yuque/__latex/679bfb0b3a85dd0534f371cb5a1ebdf0.svg)。
   2. 令导数等于 0：![img](https://cdn.nlark.com/yuque/__latex/86221c1230e0a527fe7395a3ec3ae202.svg)。
   3. 解方程，求出的 ![img](https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg) 就是极值点。

- **案例**：![img](https://cdn.nlark.com/yuque/__latex/e0ad833742a721159c703cc93e638187.svg)

- - 求导：![img](https://cdn.nlark.com/yuque/__latex/bcb464ae3aa6aaa9570d3221147a66ac.svg)
  - 令 ![img](https://cdn.nlark.com/yuque/__latex/57d185b75e595a194866fe76acbe774c.svg)。我们直接算出了最小值点。

- **为什么深度学习不用它？**

- - **参数爆炸**：神经网络可能有 **100亿个参数**，你要解一个包含 100亿个未知数的方程组。
  - **非线性复杂性**：网络中包含大量的 ReLU、Sigmoid 等非线性激活函数，导致方程组极其复杂，根本**没有解析解**（解不出来）。
  - *结论*：解析法在 AI 领域行不通。

#### (2) 数值法 (Numerical Solution) —— “盲人探路”

既然算不出精确解，数学家退而求其次：**逼近解**。

- **原理**：我不知道最小值在哪，但我知道**往哪个方向走能让函数值变小**。
- **核心**：**泰勒展开式 (Taylor Series)**。它证明了，只要沿着梯度的反方向走一小步，函数值大概率会下降。
- **地位**：这就是**梯度下降算法**的数学基石。

#### (3) 凸函数与非凸函数 (Convex vs. Non-Convex)

这是理解“最小值”的关键分类。

- **凸函数 (Convex)**：

- - **定义**：画一条线连接函数图像上的任意两点，如果这条线都在函数图像的**一方**，那就是凸函数。
  - **形状**：像一个**碗**。例如 ![img](https://cdn.nlark.com/yuque/__latex/69f1621e3cec32fa98a3ab5d08f09b75.svg)。
  - **性质**：**局部最小值 = 全局最小值** 
  - **意义**：如果你做**线性回归**（Linear Regression），Loss 是凸的，闭着眼跑梯度下降都能找到最优解。

- **非凸函数 (Non-Convex)**：

- - **形状**：像连绵起伏的**山脉**，有无数个山谷。
  - **现状**：**深度神经网络的 Loss Function 都是高度非凸的**。

#### 为什么它很重要？

- - **理想情况**：如果 Loss Function 是凸函数（比如线性回归的 MSE），那么梯度下降**一定能**找到全局最优解，闭着眼睛跑都行。
  - **现实情况 (非凸优化)**：

- - - 深度神经网络的 Loss Function 通常是**非凸的 (Non-Convex)**。
    - 地形非常复杂，有无数个**局部最小值 (Local Minima)** 和 **鞍点 (Saddle Points)**。
    - **对策**：这就是为什么我们需要动量 (Momentum)、Adam 等高级优化器，以及随机性（SGD 的震荡其实有助于跳出局部最优）。

#### ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614902648-bc13f546-804d-411d-ba28-f1de52196cf2.png)

------

### 2. 如何找到工程中的 Loss 最小值

在实际的 AI 工程中，我们的目标和数学家略有不同。数学家追求“完美的最值”，工程师追求“**好用的模型**”。

#### (1) 真正的敌人：鞍点 (Saddle Points) 而非局部最小值

以前我们认为神经网络容易卡在**局部最小值 (Local Minima)** 出不来。但最新的研究（如 Yann LeCun 等人的研究）表明，在高维空间中，真正的拦路虎是**鞍点**。

- **什么是鞍点**：

- - 形状像**马鞍**。
  - 在一个方向看是最小值（上坡），在另一个方向看是最大值（下坡）。
  - *问题*：在鞍点处，**梯度也为 0**！模型会以为自己到底了，停止更新，但其实这只是个平原。

- **工程对策**：

- - 使用**带有动量 (Momentum)** 的优化器（如 SGD+Momentum, Adam）。
  - *类比*：给下山的小球一个**惯性**。即使到了平地（鞍点），靠着惯性也能冲过去，继续往下滚。

#### (2) 并不追求“全局最小值” (Global Minimum)

- **数学视角**：必须找到 Loss 绝对最小的点。
- **工程视角**：**千万别盲目追求全局最小值！**

- - *原因*：全局最小值可能会导致**过拟合 (Overfitting)**。在深度学习的非凸优化中，平坦的局部最小值通常泛化更好，而尖锐的最小值容易过拟合。我们追求的是泛化能力强的解，而非绝对的全局最小。

- **目标**：我们要找的是一个**泛化能力强**的局部最小值（Flat Minima）。

- - *操作*：利用 **Early Stopping（早停法）**。当验证集的 Loss 不再下降时，即使训练集的 Loss 还能降，也赶紧停手。

#### (3) 逃离陷阱的工程技巧

为了在复杂的 Loss 地形中找到好结果，工程师发明了各种“外挂”：

1. **参数初始化 (Initialization)**：

- - **千万不能全设为 0**。如果 ![img](https://cdn.nlark.com/yuque/__latex/606c65cec944054450f10d90692914fb.svg)，所有神经元学到的东西都一样，网络就退化了。
  - *做法*：随机初始化（Xavier, Kaiming 初始化），把小球随机扔在半山腰，增加滚到底的几率。

1. **学习率调度 (Learning Rate Scheduler)**：

- - **Warmup**：刚开始步子小一点，热身。
  - **Decay**：中间步子大一点，快跑。
  - **Fine-tune**：最后快到底时，步子极小，精细寻找最低点。

1. **使用 Adam 优化器**：

- - BGD/SGD 是手动挡车，**Adam** 是自动驾驶。它能根据地形自动调整每个参数的学习率，是目前工程中**最常用**的 Loss 最小化工具。

### 3. 梯度下降算法的核心概念与关键要素

#### 核心概念：它是什么？

梯度下降（Gradient Descent）是神经网络的**优化引擎**。

它的核心逻辑非常简单：**通过不断地修改参数（权重** ![img](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg) **和偏置** ![img](https://cdn.nlark.com/yuque/__latex/d29c2e5f4926e5b0e9a95305650f6e54.svg)**），让损失函数（Loss）的值越来越小。**

- **公式**：

![img](https://cdn.nlark.com/yuque/__latex/bb33a8bde162c92376f6741571985102.svg)

- - ![img](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)：参数（比如权重 ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg)）。
  - ![img](https://cdn.nlark.com/yuque/__latex/7483c6745bb07f292eba02b3a9b55c26.svg)：学习率（步长）。
  - ![img](https://cdn.nlark.com/yuque/__latex/8a0ce039f865d14f8f3a22bc1e554fec.svg)：梯度（方向）。
  - **减号 (-)**：表示我们要**逆着**梯度的方向走（因为梯度指向山顶，我们要去山谷）。

------

#### 关键要素 (Key Elements)

##### (1) 梯度 (Gradient) —— “方向向导”

- **定义**：函数在某一点变化最快的方向。
- **作用**：它告诉我们，参数 ![img](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg) 应该变大还是变小，才能让 Loss 降低。
- **物理类比**：你在山上，闭上眼睛，用脚感受周围最陡峭的下坡方向。

##### (2) 学习率 (Learning Rate) —— “步长控制”

这是最重要的**超参数**。

- **定义**：每次更新参数时，步子迈多大。
- **过大**：步子太大，直接跨过谷底，甚至跑到对面山上去了（**震荡/发散**）。
- **过小**：像蚂蚁搬家，走到天黑也走不到谷底（**收敛太慢**）。

------

#### 两大拦路虎：梯度消失与梯度爆炸

这是面试和实际工程中必须掌握的概念。它们产生的原因都源于**反向传播中的链式法则（连乘效应）**。

假设一个深层网络，梯度计算公式大致如下：

![img](https://cdn.nlark.com/yuque/__latex/623c8e0bd0f4e65bb8204fdddc99597c.svg)

##### (1) 梯度消失 (Gradient Vanishing) —— “有心无力”

- **现象**：在反向传播过程中，梯度值越传越小，最后趋近于 **0**。

- **后果**：

- - **前端罢工**：靠近输出层的参数还在更新，但**靠近输入层的参数几乎不动**。
  - **变回浅层**：一个 100 层的网络，实际上只有最后几层在干活，前面的 90 多层都“死”了，根本学不到特征。

- **成因**：

- - **激活函数背锅**：使用了 **Sigmoid** 或 **Tanh**。
  - *数学解释*：Sigmoid 的导数最大只有 **0.25**。
  - ![img](https://cdn.nlark.com/yuque/__latex/576d19b9d273567c4849e99609df3042.svg)。层数越深，死得越快。

- **通俗类比：传话游戏**

- - 老板（输出层）吼了一句：“我们要彻底整改！”
  - 经理（中间层）传话声音小了一点：“我们要改。”
  - ... 传了 10 层 ...
  - 实习生（输入层）听到的是：“...（沉默）”。
  - **结果**：实习生根本不知道要改，所以工作方式完全没变。

##### (2) 梯度爆炸 (Gradient Exploding) —— “失去控制”

- **现象**：在反向传播过程中，梯度值越传越大，最后变成**无穷大 (NaN)**。

- **后果**：

- - **一步登天**：参数更新幅度过大，权重直接飞出合理范围。
  - **程序崩溃**：计算机数值溢出，训练直接报错停止。

- **成因**：

- - **初始权重过大**：如果初始权重 ![img](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg) 大于 1（比如 2）。
  - ![img](https://cdn.nlark.com/yuque/__latex/8bc00adda0a633f92af07e4788d9bfdc.svg)。
  - 这在 **RNN (循环神经网络)** 处理长文本时最容易发生。

- **通俗类比：滚雪球**

- - 山顶的一个小雪球（误差），滚下山时越滚越大，最后变成了一场雪崩，把房子（模型）都压垮了。

------

##### 4. 解决方案 (Solutions Cheat Sheet)（ps: 我也不知道里边的玩意儿是啥，先看看涨涨姿势~）

在工程中，当我们遇到这两个问题时，通常采用以下“组合拳”：

| **问题** | **解决方案**                     | **原理**                                                     |
| -------- | -------------------------------- | ------------------------------------------------------------ |
| **消失** | **使用 ReLU 激活函数**           | **(首选)** ReLU 在正区间的导数恒为 1，![img](https://cdn.nlark.com/yuque/__latex/0e6808f22a365d0d78cab24990d61c10.svg)，梯度畅通无阻。 |
| **消失** | **ResNet (残差连接)**            | 给梯度修一条“高速公路”（Shortcut），让它能跳过中间层，直接传到前面。 |
| **两者** | **Batch Normalization (BN)**     | 强制把每一层的数据拉回到标准分布，防止过大或过小。           |
| **两者** | **合理的初始化**                 | 使用 **Xavier** 或 **He Initialization**，不要随便设初始权重。 |
| **爆炸** | **梯度裁剪 (Gradient Clipping)** | 设置一个阈值（比如 5）。不管梯度多大，超过 5 就强行砍成 5。  |

### 梯度下降的三种形式

在深度学习的训练过程中，我们面临一个核心选择：**到底应该用多少数据来计算一次梯度？**

根据数据量的不同，演化出了三种主要的梯度下降策略。

#### 1. 批量梯度下降 (BGD, Batch Gradient Descent) —— “稳重的大象”

- **原理**：在每一次参数更新之前，遍历训练集中的**所有样本**（Whole Dataset）。计算所有样本梯度的平均值，然后才迈出一步。
- **公式**：

![img](https://cdn.nlark.com/yuque/__latex/3ae502d9ce9ff25d0840973e7818b95e.svg)

- **通俗类比**：**全班投票**。班长要决定去哪里春游，必须等全班 50 个人都投完票，统计完所有结果，才最终决定方向。

- **优缺点**：

- - ✅ **优点**：方向最准，一定能朝着全局最优（凸函数）或局部最优（非凸函数）收敛；曲线平滑，震荡少。
  - ❌ **缺点**：**慢！** 如果有 100 万条数据，算一次梯度要很久。而且对内存要求极高，无法进行在线学习（Online Learning）。

- ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614809635-c459ab0b-b3f1-46bd-8032-8b34a1c2c3e2.png)

#### 2. 随机梯度下降 (SGD, Stochastic Gradient Descent) —— “鲁莽的兔子”

- **原理**：每次只随机抽取**一个样本**（Single Sample），算它的梯度，立刻更新参数。
- **公式**：

![img](https://cdn.nlark.com/yuque/__latex/9d8ceebff70e00cc0223135c57a2a406.svg)

- **通俗类比**：**路人问路**。班长随便抓到一个同学问“你想去哪”，同学说“去公园”，班长立刻就往公园走一步。然后再抓下一个同学。

- **优缺点**：

- - ✅ **优点**：**快！** 立竿见影。而且因为它的随机性，有助于模型跳出局部最优解（Local Minima）。
  - ❌ **缺点**：**极不稳定**。因为单个样本可能有噪音（比如这个同学瞎指路），导致更新路线像醉汉走路一样曲折（Zigzag），很难收敛到精确的最低点，而是在最低点附近震荡。

#### 3. 小批量梯度下降 (MBGD, Mini-batch Gradient Descent) —— “明智的中庸之道”

**(工业界的默认标准)**

- **原理**：结合了 BGD 和 SGD 的优点。每次从数据集中抽取**一小批数据**（Batch Size，通常是 ![img](https://cdn.nlark.com/yuque/__latex/055ce37910d06a8239ef5a1ee87765f5.svg) ，如 32, 64, 128），算出这批数据的平均梯度，更新参数。
- **公式**：

![img](https://cdn.nlark.com/yuque/__latex/40e8660743b4f653f0458a91765273b2.svg)

- **通俗类比**：**小组讨论**。把全班分成几个小组（每组 32 人）。第一组讨论完给个方向，走一步；第二组讨论完给个方向，再走一步。

- **优缺点**：

- - ✅ **优点**：**又快又稳**。利用了 GPU 的矩阵并行计算能力（算 32 个数据的时间和算 1 个数据差不多），同时梯度方向比 SGD 准得多。
  - ❌ **缺点**：引入了一个新的超参数 **Batch Size** 需要调整。

- ![img](https://cdn.nlark.com/yuque/0/2026/png/35927690/1769614863799-cf36634f-5b4e-46cb-9a80-8d896d6abbcc.png)

------

#### 4. 核心概念辨析：Epoch, Batch, Iteration **(补充说明)**

初学者最容易晕的三个词，这里一次性讲清楚。

假设我们有 **1000** 个数据样本，设置 **Batch Size = 100**。

1. **Batch Size (批大小)**：

- - 每次喂给模型的数据量。
  - *本例*：100。

1. **Iteration (迭代次数)**：

- - 更新一次参数叫一次 Iteration。
  - *计算*：要把 1000 个数据都过一遍，需要分 ![img](https://cdn.nlark.com/yuque/__latex/5691c9f702d9b9f064b9eb6b4203f8ed.svg) 批。
  - 所以，完成一轮训练需要 **10 个 Iterations**。

1. **Epoch (轮数)**：

- - 把**所有**训练数据（1000个）都学过一遍，叫一个 Epoch。
  - *本例*：1 Epoch = 10 Iterations。
  - 通常我们需要几十甚至上百个 Epoch 才能把模型练好。

------

#### 5. 三种算法的对比总结 (Summary Table)

| **特性**        | **BGD (批量)**     | **SGD (随机)**     | **MBGD (小批量)**        |
| --------------- | ------------------ | ------------------ | ------------------------ |
| **数据量/次**   | 全部数据 (N)       | 1 个样本           | Batch Size (如 64)       |
| **计算速度/步** | 极慢               | 极快               | 快 (GPU加速)             |
| **收敛路径**    | 平滑直线，直奔谷底 | 剧烈震荡，曲折前行 | 略带震荡，整体稳定       |
| **收敛位置**    | 极值点             | 极值点附近徘徊     | 极值点附近 (较 SGD 更近) |
| **内存占用**    | 极大 (可能 OOM)    | 极小               | 适中 (可控)              |
| **应用场景**    | 小数据理论研究     | 在线学习           | **绝大多数深度学习任务** |



#### 6. Python 代码示例

这段代码模拟了“下山”的过程。

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义一个函数 (假设这是 Loss Function: y = x^2)
# 这是一个像碗一样的函数，最低点在 x=0
def loss_function(x):
    return x ** 2

# 2. 定义导数 (梯度: y' = 2x)
def gradient(x):
    return 2 * x

# 3. 梯度下降过程
def train(start_x, learning_rate, steps):
    x = start_x
    history = [x]

    for i in range(steps):
        grad = gradient(x)                # 1. 感觉坡度
        x = x - learning_rate * grad      # 2. 向下走一步 (核心公式)
        history.append(x)
        print(f"Step {i+1}: x = {x:.4f}, grad = {grad:.4f}")

    return history

# --- 运行模拟 ---
# 从 x=5 的位置开始下山，步长 0.1
path = train(start_x=5, learning_rate=0.1, steps=10)

# --- 绘图展示 ---
# 可以看到点一步步逼近 0
x_vals = np.linspace(-6, 6, 100)
plt.plot(x_vals, loss_function(x_vals), label="Loss Function")
plt.scatter(path, [loss_function(p) for p in path], color='red', label="Steps")
plt.title("Gradient Descent Visualization")
plt.legend()
plt.show()
```

------

## 卷积神经网络 (CNN)：视觉处理的基石

### 1. CNN 特征介绍与结构**(Features & Architecture)**

#### 为什么要发明 CNN？(全连接层的痛点)

假如我们有一张 ![img](https://cdn.nlark.com/yuque/__latex/3dc3449846f99d13e5d7516dd9cee455.svg) 像素的彩色图片（RGB 3通道）。

- **输入维度**：![img](https://cdn.nlark.com/yuque/__latex/d99057fadb8be4ca3ced894c5361cea9.svg) 个像素点。
- **全连接层**：如果隐藏层只有 1000 个神经元。
- **参数量**：![img](https://cdn.nlark.com/yuque/__latex/ba040496bdada160778fc1d707e7fc73.svg) **亿**个权重参数！
- **结果**：计算机内存瞬间爆炸，根本没法训练。而且全连接层会破坏像素之间的**空间相邻关系**（把二维图片拉成了一维长条）。

#### CNN 的两大核心特性

CNN 通过以下两个特性，完美解决了上述问题：

1. **局部感知 (Local Connectivity)**：

- - 神经元不需要看整张图，只需要看**一小块区域**（比如 ![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg) 的像素）。这就像人眼看东西，先聚焦局部细节。

1. **参数共享 (Parameter Sharing)**：

- - **核心思想**：用来识别“左上角有没有鸟嘴”的探测器（卷积核），同样可以拿来探测“右下角有没有鸟嘴”。
  - 不管鸟飞到图片的哪里，特征是一样的。这大大减少了需要训练的参数数量。

------

### 2. 卷积层原理 (Convolutional Layer) —— “特征提取器”

这是 CNN 最核心的部分。它的任务是**提取图片中的特征**（如边缘、纹理、形状）。

#### 1. 核心逻辑：从“全看”到“扫描”

- **全连接层 (FC) 的逻辑**：

- - **方式**：一眼看全图。
  - **缺点**：对于一张 ![img](https://cdn.nlark.com/yuque/__latex/3dc3449846f99d13e5d7516dd9cee455.svg) 的图片，如果每个像素都连一个权重，参数量是天文数字。且它破坏了像素间的**邻域关系**（比如眼睛和眉毛是挨着的，拉直后就分开了）。

- **卷积层 (Conv) 的逻辑**：

- - **方式**：**局部扫描**。用一个小的“窗口”在图片上滑来滑去。
  - **假设**：图片中某个局部的特征（比如鸟嘴），无论出现在图片的左上角还是右下角，它的特征是一样的。所以我们可以**共用同一个探测器**（参数共享）。

------

#### 2. 内部结构：卷积核 (Filter / Kernel)

卷积层的核心组件就是**卷积核**。

- **本质**：它是一个小的权重矩阵（Weight Matrix），通常是 ![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg) 或 ![img](https://cdn.nlark.com/yuque/__latex/fb6eedf35034ac23e8645a3b84d70168.svg)。
- **功能**：你可以把它想象成一个**“特征过滤器”**。

- - 有的核专门过滤**横线**。
  - 有的核专门过滤**竖线**。
  - 有的核专门过滤**颜色**。

##### 案例：垂直边缘检测

假设我们有一个专门检测“垂直边缘”的卷积核：

![img](https://cdn.nlark.com/yuque/__latex/4204ff867054733657060b0b826ccbc1.svg)

- **当它扫到纯色区域（无边缘）**：比如全是 10 的像素。![img](https://cdn.nlark.com/yuque/__latex/2d7abf14f496f296b3230a71c32ce4f2.svg)。结果为 0，表示“没发现边缘”。
- **当它扫到边缘区域**：左边是 10，右边是 0。![img](https://cdn.nlark.com/yuque/__latex/980eb803634385bbbdb79182c07a297d.svg)。结果很大，表示“发现边缘”！

------

#### 3. 卷积的数学运算 (The Math Calculation)

这是最容易混淆的地方。虽然名字叫“卷积 (Convolution)”，但在深度学习中，实际操作其实是**互相关 (Cross-Correlation)**，简单说就是**“对应位置相乘再求和”**。

假设：

- **输入 (Input)**：![img](https://cdn.nlark.com/yuque/__latex/fb6eedf35034ac23e8645a3b84d70168.svg) 图像。
- **卷积核 (Kernel)**：![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg)。

**计算步骤**：

1. **对齐**：将 ![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg) 的核盖在图像的某个区域上。
2. **点积 (Element-wise Product)**：核里的 9 个数，和图像上被盖住的 9 个数，**对应位置分别相乘**。
3. **求和 (Sum)**：将这 9 个乘积加起来，得到**一个标量值**。
4. **填入**：将这个值填入输出矩阵（Feature Map）的对应位置。
5. **滑动**：向右移动一格，重复上述步骤。

**【手算演示】**

![img](https://cdn.nlark.com/yuque/__latex/978f67ad6c34007c7670ddca6c96f807.svg)

![img](https://cdn.nlark.com/yuque/__latex/1210f7d9ab1ce1790f66160b8749f2ed.svg)

**【图片演示】**

![img](https://cdn.nlark.com/yuque/0/2026/gif/35927690/1769614301898-7a55a001-0312-42ff-b486-3d61fb03f01c.gif)

------

#### 4. 三大关键参数 (Key Parameters)

在写代码（如 PyTorch 的 `nn.Conv2d`）时，你需要设置三个关键参数：

##### (1) 核尺寸 (Kernel Size)

- 窗口有多大。最常用的是 ![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg)。
- *为什么选 3x3？* 它是能捕获“中心+邻居”信息的最小奇数尺寸，堆叠多个 3x3 可以达到大尺寸核的效果，但参数更少。

##### (2) 步长 (Stride)

- **定义**：窗口每次滑动的距离。
- **Stride = 1**：地毯式搜索，输出尺寸变化不大。
- **Stride = 2**：跳着搜索，**输出尺寸直接减半**（起到了类似池化的降维作用）。

##### (3) 填充 (Padding)

- **问题**：如果不填充，每次卷积后图片都会变小（![img](https://cdn.nlark.com/yuque/__latex/fb6eedf35034ac23e8645a3b84d70168.svg) 卷完变成 ![img](https://cdn.nlark.com/yuque/__latex/b08d861763b277c3ff6c7f92bc62b06c.svg)）。而且图片**边缘的像素**只被扫到一次，信息丢失严重。
- **解决**：在图片周围**补一圈 0**。
- **Same Padding**：补完后，使得输出尺寸 = 输入尺寸。

------

#### 5. 输出尺寸计算公式 (The Formula)

这是考试和工程中经常要算的：经过一层卷积后，我的图片变成多大了？

假设：

- 输入尺寸：![img](https://cdn.nlark.com/yuque/__latex/d9bf7d083e6c093ca6e269a20c34be0e.svg)
- 卷积核：![img](https://cdn.nlark.com/yuque/__latex/0bc5c73f7b824a9200254c54edc22e8e.svg) (Kernel Size)
- 填充：![img](https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg) (Padding)
- 步长：![img](https://cdn.nlark.com/yuque/__latex/55fc237afbe535f7d8434985b848a6a7.svg) (Stride)

**输出宽度** ![img](https://cdn.nlark.com/yuque/__latex/b625028757171dc8d66e1c8adf1ed223.svg) **计算公式**：

![img](https://cdn.nlark.com/yuque/__latex/9308cf55688c7790eef964727455c034.svg)

*(注：如果除不尽，通常向下取整)*

```python
import torch
import torch.nn as nn

# 1. 定义一个模拟图片
# 格式: [Batch_Size, Channel, Height, Width]
# 假设: 1张图片, 3个通道(RGB), 大小 32x32
input_img = torch.randn(1, 3, 32, 32)

# 2. 定义卷积层
# 输入通道 3 -> 输出通道 16
# 卷积核 3x3, 步长(stride)=2, 填充(padding)=1
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, 
                       kernel_size=3, stride=2, padding=1)

# 3. 前向传播
output = conv_layer(input_img)

# 4. 验证尺寸
print(f"输入尺寸: {input_img.shape}")  # [1, 3, 32, 32]
print(f"输出尺寸: {output.shape}")     # [1, 16, 16, 16]

# --- 验证公式 ---
# W_out = (W_in - F + 2P) / S + 1
#       = (32 - 3 + 2*1) / 2 + 1
#       = 31 / 2 + 1 = 15.5 + 1 -> 向下取整 15 + 1 = 16
# 结果一致！
```

------

#### 6. 多通道卷积 (Multi-Channel Convolution)

我们处理的通常是彩色图片（RGB 3通道）。这时候卷积怎么做？

- **立体卷积**：如果输入是 ![img](https://cdn.nlark.com/yuque/__latex/0599b8a6f75373cb7f090d88ca5b672a.svg)，那么卷积核**不再是二维的纸片**，而是一个**立方体**，尺寸为 ![img](https://cdn.nlark.com/yuque/__latex/1498391b25046d2117b6cd570f6a4b42.svg)。
- **一次计算**：这个立方体卷积核会同时处理 3 个通道，计算出**一个**值（不是 3 个）。
- **多个核**：如果我们想要提取 64 种特征，我们就用 **64 个** 这样的立方体卷积核。
- **最终输出**：![img](https://cdn.nlark.com/yuque/__latex/8e6e7f27a637957438483761c8af6b2a.svg)（64 个特征图叠在一起）。

------

### 3. 池化层 (Pooling Layer) —— 特征压缩机

卷积之后的数据量往往还是太大，而且我们不需要知道特征精确到哪个像素点（比如我们只需要知道“图里有只鸟”，不需要知道鸟嘴精确在坐标 (100.5, 20.1)）。

#### (1) 最大池化 (Max Pooling) —— **最常用**

- **原理**：在选定区域内（如 ![img](https://cdn.nlark.com/yuque/__latex/829db47d500df094775efa8c62bdd9ea.svg)），**只保留最大的那个数**，丢弃其他三个。
- **举例**：

![img](https://cdn.nlark.com/yuque/__latex/6947a4390a2760458d827a1bc5298b66.svg)

- **作用**：提取最显著的特征（谁的值大，谁最重要），同时去除噪音背景。

#### (2) 平均池化 (Average Pooling)

- **原理**：计算区域内的平均值。
- **举例**：

![img](https://cdn.nlark.com/yuque/__latex/07c76bb6935bc66ab0d061e2c1532749.svg)

- **作用**：保留背景信息，使图像更平滑（现在用的比较少了）。

#### (3) 为什么需要池化？

- **降维**：直接减少参数数量，加快计算速度。
- **防过拟合**：丢弃了一部分不重要的信息，让模型更鲁棒。
- **平移不变性**：稍微移动一下物体，最大值依然是那个最大值，输出结果不变。

------

### 4. 全连接层 (Fully Connected Layer) —— 分类决策大脑

经过前面几十层的卷积和池化，我们提取到了高度抽象的特征（比如“有耳朵”、“有尾巴”、“有毛”）。全连接层负责把这些线索串起来，给出最终答案。

#### (1) 扁平化 (Flatten)

- 卷积出来的特征图是立体的（比如 ![img](https://cdn.nlark.com/yuque/__latex/215b50d631adfadbfba6e61adfb729ff.svg)）。
- 全连接层只能吃一维向量。
- 所以第一步是把**立体积木推倒**，拉成一条长长的向量（![img](https://cdn.nlark.com/yuque/__latex/39484cc3485bb0a17a38186d9913d3e0.svg) 个点）。

#### (2) 逻辑推断

- 这部分就是我们最开始学的**标准神经网络 (MLP)**。
- 每个神经元都和上一层的所有特征相连。
- 它通过权重计算，判断这些特征组合起来属于哪个类别。

- - *比如*：权重发现“圆耳朵”和“长胡须”同时出现，就会给“猫”这个类别投高分。

#### (3) 输出 (Softmax)

- 最后接一个 Softmax 激活函数，把分数变成概率。
- *Output*: [猫: 0.8, 狗: 0.1, 猪: 0.1]。

------

### 5. 总结：CNN 的完整工作流

| **层次**        | **就像是...**   | **核心功能**              | **输入/输出变化**                                            |
| --------------- | --------------- | ------------------------- | ------------------------------------------------------------ |
| **Input**       | 眼睛            | 接收原始像素              | 原始图片                                                     |
| **Convolution** | **显微镜/滤镜** | **提取特征** (线条、纹理) | 图片变厚 (通道数增加)                                        |
| **ReLU**        | 开关            | 引入非线性                | 数据不变                                                     |
| **Pooling**     | **缩略图工具**  | **压缩数据** (降维)       | 图片变小 (长宽减半)                                          |
| **FC Layer**    | **大脑**        | **综合判断** (分类)       | 3D矩阵 ![img](https://cdn.nlark.com/yuque/__latex/33b44e34aa35b8c4ecd0606453ee68e9.svg) 概率向量 |

------

## 深度学习的大致流程

### 1. 数据预处理 (Data Processing) —— 从“表格”到“张量”

在 ML 中，你处理的通常是 Excel 表格（结构化数据）。而在 DL 中，我们更多处理**非结构化数据**（图片、文本、声音）。

- **核心任务**：把一切变成 **张量 (Tensor)**。

- - 神经网络不认识图片，也不认识汉字，它只认识数字矩阵。
  - **图片**：变成 `[通道数, 高, 宽]` 的像素矩阵（比如 `3x224x224`）。
  - **文本**：变成 `[句子长度, 词向量维度]` 的数值矩阵。

- **代码关键词**：`Dataset`, `DataLoader` (负责把数据打包喂给模型)。

### 2. 定义网络结构 (Model Design) —— “搭积木”

这是 DL 与 ML **最大的不同**。

- **ML**：你直接选一个模型（如“随机森林”），它是现成的黑盒。
- **DL**：你需要自己**设计**模型。就像搭积木一样，决定第一层用什么，第二层用什么，一共多少层。
- **常用组件**：

- - 全连接层 (Linear)
  - 卷积层 (Conv2d) - 处理图片
  - 激活函数 (ReLU, Sigmoid)

**代码示例**：

```python
# 这种层层堆叠的感觉
model = nn.Sequential(
    nn.Linear(784, 128),  # 第一层
    nn.ReLU(),            # 激活
    nn.Linear(128, 10)    # 输出层
)
```

### 3. 定义“裁判”与“教练” (Loss & Optimizer)

在 ML 中，这些通常封装在算法里了。但在 DL 中，你需要显式指定：

- **损失函数 (Loss Function)**：即“裁判”。用来衡量模型预测得有多烂（Loss 越小越好）。

- - 比如分类任务用 `CrossEntropyLoss`，回归任务用 `MSELoss`。

- **优化器 (Optimizer)**：即“教练”。用来告诉模型如何根据 Loss 来修改参数（梯度下降）。

- - 比如 `SGD` (随机梯度下降) 或 `Adam`。

### 4. 训练循环 (Training Loop) —— 手动挡的核心

在 ML 中，这一步就是一行 `model.fit(X, y)`。 但在 DL 中，你需要写一个 `for` 循环，**显式地**写出每一步的更新逻辑（反向传播）。

**标准 4 步走（这也是面试必问的流程）：**

1. **Forward (前向传播)**：模型猜结果。
2. **Loss (计算误差)**：裁判打分。
3. **Backward (反向传播)**：计算梯度（找原因）。
4. **Step (更新参数)**：优化器修改权重。

```python
# 伪代码感受一下手动挡
for epoch in range(100):      # 练 100 轮
    pred = model(data)        # 1. 猜
    loss = loss_fn(pred, target) # 2. 算误差
    loss.backward()           # 3. 算梯度
    optimizer.step()          # 4. 改参数
    optimizer.zero_grad()     # 清空梯度准备下一轮
```

### 5. 模型评估 (Evaluation)

这一步和 ML 基本一样。用准确率 (Accuracy)、召回率 (Recall) 等指标在**测试集**上验证模型效果。