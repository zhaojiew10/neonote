模型转换，就是当个翻译官，让不同语言的模型能在同一个场景里交流。现在训练模型，大家习惯用PyTorch、TensorFlow这些，它们功能强大，但更侧重于研究和创新，比如搞分布式训练、自动微分。但到了实际应用，比如手机、服务器上跑模型，就得靠推理框架，像TensorRT、OpenVINO，它们更擅长把模型塞进硬件加速，跑得快、占资源少。

问题是，训练框架和推理框架，它们的模型表示方式、内部语言都不一样。所以，模型转换就成了那个关键的桥梁，把训练好的模型，翻译成推理框架能懂的格式，才能顺利部署到实际应用中。

有了模型转换，我们还需要一个强大的推理引擎来驱动整个系统。你可以把它想象成AI应用的发动机，负责把模型加载起来，然后让它跑起来。这个推理引擎主要分两块：优化和运行。

- 优化阶段，包括各种模型转换工具，比如把各种格式的模型转换成统一的中间表示，再进行各种优化，还有模型压缩，甚至端侧学习的组件。
- 运行阶段，负责模型的加载和执行。

<img src="assets/image-20250505173625227.png" alt="image-20250505173625227" style="zoom:50%;" />

其中模型转换工具模块。它主要干两件事：

- 把不同框架的模型格式，比如PyTorch的pt文件、TensorFlow的pb文件，转换成推理引擎自己能理解的中间表示，也就是IR。
- 计算图优化。拿到模型的计算图之后，不是直接用，而是要进行优化。比如，把连续的几个操作，比如卷积加ReLU激活，合并成一个更高效的融合操作，或者调整数据的存储顺序，让它更适合GPU的计算习惯。

## 转换和优化的挑战

这个转换过程可不是一帆风顺的，有如下两个挑战

（1）算子的统一

|    框架    |   导出方式   | 导出成功率 | 算子数（不完全统计） | 冗余度 |
| :--------: | :----------: | :--------: | :------------------: | :----: |
|   Caffe    |    Caffe     |     高     |          52          |   低   |
| TensorFlow |     I.X      |     高     |         1566         |   高   |
|            |    Tflite    |     中     |         141          |   低   |
|            |     Self     |     中     |        1200-         |   高   |
|  Pytorch   |     Onnx     |     中     |         165          |   低   |
|            | TorchScripts |     高     |         566          |   高   |

神经网络模型本质上就是一堆算子，比如卷积、池化、ReLU等等。但问题是，不同框架，像PyTorch、TensorFlow、Caffe，它们定义的算子，虽然名字可能差不多，但实现细节、行为模式可能完全不一样。比如，PyTorch的Padding和TensorFlow的Padding，填充方式、方向都不一样。PyTorch的卷积层padding可以随便指定，TensorFlow的不行，得用tf.pad函数。所以，推理引擎自己定义一套标准的算子集，然后用这套标准去适配不同的框架，实现所谓的算子统一。

（2）模型文件格式的多样性

| **AI 框架**  |         **模型文件格式**         |
| :----------: | :------------------------------: |
|   PyTorch    |            .pt, .pth             |
|  MindSpore   |   .ckpt, .mindir, .air, .onnx    |
| PaddlePaddle |   .pdparams, .pdopt, .pdmodel    |
|  TensorFlow  | .pb(Protocol Buffers), .h5(HDF5) |
|    Keras     |           .h5, .keras            |

主流的框架，PyTorch、MindSpore、Paddle、TensorFlow、Keras，它们导出的模型文件格式，像pt、pth、ckpt、mindir、pdparams、pb、h5、keras，五花八门。而且，同一个框架，不同版本，比如TensorFlow 1.x和2.x，里面的算子也可能不一样。这就像一个巨大的迷宫，每种格式都是一个出口，你需要找到对应的路径才能进去。要解决这个问题，还是得靠推理引擎的通用能力。它需要支持自定义的计算图IR，能够把各种格式的模型都转换成这个统一的IR，然后再进行后续的推理。

（3）**网络结构的多样性**。现在流行的网络结构，像CNN、RNN、Transformer，擅长处理的任务也不同。比如CNN擅长图像，RNN擅长序列，Transformer是NLP的王者。推理引擎得能支持各种主流网络结构，还得能跑得动，跑得快。这就需要推理引擎提供丰富的Demo和Benchmark，告诉大家怎么用它来处理不同类型的网络，比如TensorRT的Demos，还有MLPerf这样的基准测试。

（4）**输入输出的复杂性**。现在的神经网络，输入输出形式五花八门，可能有多个输入，多个输出，维度可能不固定，甚至输入的形状在运行时会动态变化。还有带控制流的模型，比如有if-else判断的。这给推理引擎提出了很高的要求。引擎必须足够灵活，能适应各种输入输出形式，还得具备处理动态形状的能力。

再来看看优化的目标。优化的核心目标，就是消除冗余，让模型跑得更快、更省资源。这里主要针对四种冗余：**结构冗余、精度冗余、算法冗余和读写冗余**。

- 结构冗余，就是模型里那些没用的计算或者重复的计算，比如训练时可能留下的无效节点，或者模型里重复出现的模块。我们可以通过算子融合、算子替换、常量折叠来消除。
- 精度冗余，就是模型用的FP32浮点数，很多时候精度太高，浪费了计算资源。我们可以用量化，把FP32压缩到FP16甚至INT8，或者用剪枝、蒸馏这些方法来减小模型体积。
- 算法冗余，是指某些算子的实现算法本身有冗余，比如两种不同的滑动窗口操作，底层逻辑可能一样。我们可以用统一的高性能算子库，或者算子融合来解决。
- 读写冗余，就是内存访问不高效，比如重复读写数据，或者访问模式不连续导致缓存命中率低。我们可以通过数据排布优化、内存分配优化来改善。

## 转换模块的架构

我们再来看看转换模块的架构

![转换模块架构](https://chenzomi12.github.io/_images/01Introduction016.png)

它主要由两部分组成：格式转换和图优化。

- 格式转换，顾名思义，就是负责跟各种AI训练框架打交道的，比如针对MindSpore的转换器，针对TensorFlow的转换器，针对PyTorch的，还有针对PaddlePaddle的。它们的作用是把不同框架的模型转换成推理引擎统一的中间表示IR。
- 图优化阶段。这里会进行一系列的优化操作，比如算子融合、算子替换、布局调整、内存分配等等。整个流程就是：先通过前端转换器把模型转换成IR，然后在后端进行各种优化，最终得到一个更高效、更易部署的模型。这个优化过程，通常不是一步到位的，而是分阶段进行的。我们可以把它看作是三步走：Pre Optimize、Optimize 和 Pos Optimize。
  - Pre Optimize阶段，主要是做基础的语法检查和初步清理，比如消除公共表达式、死代码，做一些简单的代数简化，确保计算图的结构是基本正确的。
  - Optimize阶段，就是核心的算子优化，包括我们前面提到的算子融合、算子替换、常量折叠。
  - Pos Optimize阶段，主要关注内存和数据访问效率，比如调整数据格式，优化内存布局，合并重复的算子，减少不必要的内存访问。


## 推理文件格式

训练好模型，总不能一直放在内存里，那太浪费了。我们需要把模型保存下来，方便以后加载使用。这个过程就叫做模型序列化。反过来，从硬盘加载回内存的过程，就叫做反序列化。**序列化和反序列化，目的是让模型能够长久保存，随时可以被调用**。序列化的方法有很多种，我们可以大致分成几类。

- 跨平台跨语言通用的，像XML、JSON、Protocol Buffers、FlatBuffer。其中最常用的是Protocol Buffers，简称Protobuf，Google开发的。它是一种二进制格式，效率高，很多框架都用它，比如ONNX就是用Protobuf来存储模型的。还有苹果的CoreML，也用了Protobuf。
- 模型自身提供的序列化方法，比如TensorFlow的Checkpoint文件，PyTorch的pt和pth文件。这些是特定框架的格式，通常效率很高，但可能跨平台性差一些。
- 语言级别的通用序列化，比如Python的pickle、joblib，R语言的rda。这些是语言自带的工具，方便，但可能不够高效。
- 用户自定义的序列化方法，这通常是为了满足一些特殊的需求，比如极致的性能优化或者特定的嵌入式环境。

选择哪种方法，要看具体的应用场景和需求。以PyTorch为例，它提供了两种主要的序列化方式。

- PyTorch内部的格式，主要用torch.save和torch.load来实现。这种方式比较简单，它只保存模型的state_dict，也就是权重和偏置这些参数，以及优化器的状态，但不包含网络的计算图结构。它**底层是用Python的pickle来序列化的**。需要注意的是，如果模型是在GPU上训练的，用torch.load加载到CPU上时可能会有问题，需要用.to('cpu')显式转换。

```py
# state_dict只是一个 Python 字典对象，它将每个层映射到其参数张量
torch.save(model.state_dict(), PATH) # torch.save 将序列化对象保存到磁盘

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH)) # torch.nn.Module.load_state_dict 使用反序列化的 state_dict 加载模型的参数字典 
model.eval()
# 优化器对象 torch.optim 还有一个state_dict，其中包含有关优化器状态以及所使用的超参数的信息
```

- 通过ONNX导出。PyTorch提供了torch.onnx.export函数，可以直接把模型导出成ONNX格式。ONNX的好处是它是一个开放的、跨平台的格式，可以被不同的推理框架支持，比如TensorRT、OpenVINO、ONNX Runtime等等。这样，PyTorch训练的模型就可以方便地部署到各种平台上。**ONNX内部也是用Protobuf来存储数据的**。

```py
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names) # 导出到名为alexnet.onnx的 ONNX 文件

model = onnx.load("alexnet.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
```

下面介绍 Protobuf 和 FlatBuffers 两种流行的目标文件格式。

Protobuf 采用的是TLV结构，也就是标签-长度-值。这里的**Tag包含了字段的编号和类型，Length表示值的长度，Value就是实际的数据**。对于可变长的数据，比如字符串，需要长度；对于定长数据，比如整数，长度就省略了。

Tag的编码方式叫Varint，它能高效地编码整数。编码过程就是先构建好消息，然后逐个字段，先编码Tag、Length、Value，再把它们组合起来。解码过程反过来，先读取Tag，确定字段类型和编号，再读取Length，最后读取Value。如果Value是嵌套的Message，就递归解析。这种结构的好处是灵活、高效，能很好地适应各种复杂的数据结构。

除了Protobuf，还有一种非常流行的序列化库，叫做FlatBuffers。它也是Google开发的，开源的，跨平台。它跟Protobuf很像，但有一个很重要的区别：它**不需要解析。你序列化得到一个二进制buffer，然后可以直接访问这个buffer里的数据，不需要像Protobuf那样，先解码**，再访问。这意味着什么？意味着访问速度更快，内存占用更小，因为不需要额外的内存分配。而且，生成的代码量也更少，只需要一个头文件。它支持的语言也非常广泛，比如C++、JavaScript、Python、Rust等等。很多轻量级的推理框架，比如阿里巴巴的MNN，还有华为的MindSpore Lite，都采用了FlatBuffers。它的定义方式也很灵活，用schema文件描述数据结构，然后生成代码。对于那些对性能要求极高的场景，FlatBuffers是一个非常有吸引力的选择。

最后，我们来对比一下Protobuf和FlatBuffers。从支持的语言来看，FlatBuffers更广泛，特别是JavaScript。版本方面，Protobuf有2.x和3.x，不兼容，FlatBuffers是1.x。协议文件分别是.proto和.fbs。代码生成方面，FlatBuffers生成的代码量通常更小。字段类型支持上，两者都支持基本类型，但FlatBuffers更侧重于底层的整数类型，比如int8、uint8等，这在处理原始数据时很有优势。总的来说

- Protobuf历史悠久，生态完善，功能全面，但需要解析；
- FlatBuffers则更注重性能，特别是访问速度，不需要解析，但可能在某些语言支持或字段类型上略有差异

## 自定义计算图 IR

模型转换，本质上是对模型的结构和参数进行一次彻底的重新定义。这可不是简单的复制粘贴，而是需要深入理解模型的计算图结构，然后根据目标格式的要求，可能需要添加、删除、甚至修改节点和边。

目标是确保转换后的计算图，依然能够准确地描述原始模型的计算逻辑，不能丢信息，也不能添乱。**核心在于中间表示IR**。它就像一个万能插座，把各种AI训练框架比如MindSpore、TensorFlow、PyTorch、PaddlePaddle产生的模型文件，通过各自的转换器，统一接入到这个IR接口。

![基于计算图的图优化](https://chenzomi12.github.io/_images/01Introduction025.png)

有了这个统一的中间件，我们就可以进行各种图优化操作了，比如算子融合、替换、调整内存布局等等。这大大简化了跨平台部署和优化的复杂度。拿到统一的IR之后，真正的优化工作就开始了。目标很明确：让模型跑得更快、更省资源。这里有很多优化技巧，算子融合，就是把两个相邻的、可以合并计算的算子，比如一个加法和一个乘法，打包成一个更高效的算子，减少中间数据的搬运和存储，这就像流水线作业，效率更高。常量折叠，如果某个输入是常数，那就可以在编译时就把它计算出来，运行时就直接用结果，省去了计算时间。

现在我们来解构一下计算图本身。它由两部分构成：张量和算子。

- 张量，你可以把它想象成神经网络里的数据容器，用来承载各种信息。它的**维度叫做秩**，秩为零的就是一个标量，比如一个数字5；秩为1的就是向量，像一条线；秩为2就是矩阵，像一张纸；我们常见的RGB图像，就是三个通道，所以是秩为3的张量。张量里装的元素可以是整数、浮点数等等。这个形状Shape，就是描述张量大小的，比如3x2x5，表示这个张量有3个维度，分别是2、5、3个元素。

![张量](https://chenzomi12.github.io/_images/03IR01.png)

- 算子，它是计算图的基本运算单元，负责对张量进行各种加工处理。从简单的加减乘除，到复杂的卷积、池化、激活函数，再到控制数据流向的If-Else，应有尽有。一个算子通常会接受多个输入张量，经过一系列计算，产生一个或多个输出张量，就像流水线上的工序，一环扣一环。

![算子](https://chenzomi12.github.io/_images/03IR02.png)

在AI训练框架，比如TensorFlow、PyTorch里，计算图扮演着非常重要的角色。它不仅是模型的蓝图，更是实现自动微分梯度计算的基石。框架通过构建计算图，清晰地描述了数据是如何一步步从输入流向输出的，以及每个算子之间的依赖关系。这使得框架能够自动计算梯度，进行反向传播，从而实现模型的训练。同时，框架也会利用计算图来优化模型的执行效率，比如进行算子融合、内存优化等。可以说，计算图是AI框架高效运行和灵活开发的核心支撑。

当模型训练完成，进入部署阶段，推理引擎就登场了。推理引擎的作用是高效地执行模型，生成预测结果。在这个过程中，计算图同样至关重要。它首先被转换为一个中间表示IR，这个IR的好处是抽象，它不依赖于特定的硬件平台，比如CPU、GPU、VPU等，这样模型就可以在不同平台上部署。推理引擎会利用这个IR，进行更深层次的优化，比如前面提到的算子融合，还会精确管理内存，避免不必要的拷贝和碎片，以及进行任务调度，充分利用硬件资源，最终目标是让模型在推理时又快又稳。

虽然都是计算图，但AI框架和推理引擎的侧重点完全不同。

![AI 框架计算图 vs 推理引擎计算图](https://chenzomi12.github.io/_images/03IR03.png)

- 训练框架，比如TensorFlow，既要支持正向传播，也要支持反向传播，因为要训练模型；而推理引擎，只需要正向传播，因为预测结果就够了。

- 训练框架为了灵活性，可能支持动态图，而推理引擎为了优化性能，更倾向于静态图。

- 训练时，我们可能需要分布式并行来加速训练，但推理时，通常关注单卡或少量多卡的低延迟服务。
- 训练框架更关注算法的创新和精度，而推理引擎则更关注效率和性能。

那么，如何在自己的推理引擎中自定义一个高效的计算图呢？我们可以用Protobuf或FlatBuffers来构建一个自定义的中间表示IR。这个过程大致分三步：

1. 定义你的IR长什么样，包括张量、算子、网络结构等，要根据你的引擎特性来设计。
2. 用解析库把训练好的模型，比如TensorFlow的模型，读进来，解析成你的IR对象。
3. 把解析后的信息，用你的API导出成最终的IR文件，同时在导出过程中，就可以进行各种优化，比如算子融合、内存布局调整等等。

具体到张量的表示，我们需要定义几个关键属性。

- 数据类型，比如是浮点数、整数还是字符串？这需要一个枚举类型来定义。
- 数据内存排布格式，也就是数据在内存里是怎么存放的，比如常见的NCHW、NHWC，还有针对特定硬件优化的格式，比如NC4HW4。这些格式决定了内存访问效率。
- 张量本身，可以用一个结构体来表示，包含维度、数据格式、数据类型等信息。这样，我们就有了描述张量的标准化语言。

算子的定义则更复杂一些。因为同一个算子，在不同的AI框架里，比如TensorFlow和PyTorch，实现细节可能不一样。所以，我们需要为每个算子都定义一个独立的类型。这个类型通常包括算子的类型，比如卷积、池化、矩阵乘法，以及算子的参数，比如卷积核大小、填充方式等。为了方便管理，我们通常会用一个联合体来包含所有可能的算子参数，然后用一个枚举类型来区分不同的算子类型。这样，每个算子都有一个明确的定义和参数。

定义了张量和算子，我们还需要定义整个网络的结构。这通常用一个**网络模型Net结构来表示。它会包含网络的名称、输入输出张量的名称、以及最重要的算子列表**。这个算子列表就是网络的执行顺序，告诉推理引擎先执行哪个算子，再执行哪个。如果网络结构复杂，比如有循环或分支，我们还可以引入子图SubGraph的概念，用类似的方式定义子图的输入输出和内部算子。

## 模型转换流程

主流的转换方式有两种：

- 直接转换，就是从A框架直接转换到B框架，比如MindSpore直接转成Inference IR。
- 规范式转换，就是先转成一个大家都认可的通用格式，比如ONNX，然后再从ONNX转到目标框架，比如PyTorch转ONNX，再转成Inference IR。

这两种方式，主流框架都支持。直接转换的流程相对直接

1. 第一步是读取，把源框架的模型文件读进来，然后仔细分析里面的内容，比如张量是什么类型、什么格式，用了哪些算子、参数是多少，网络结构是怎么样的，等等。
2. 第二步是格式转换，把这些信息直接翻译成目标框架能懂的格式。如果遇到比较复杂的算子，可能需要专门写代码来处理。
3. 第三步是保存，把转换好的模型，按照目标框架的格式，保存下来。

这样就完成了从一个框架到另一个框架的直接迁移。这里有个简单的代码示例，展示了如何将一个简单的TensorFlow模型转换为PyTorch模型。

```py
import TensorFlow as tf
import torch
import torch.nn as nn

# 定义一个简单的 TensorFlow 模型
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 1. 内容读取
# 创建并训练一个简单的 TensorFlow 模型
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
tf_model = SimpleModel()
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(x_train, y_train, epochs=5)

# 2. 格式转换
# 定义对应的 PyTorch 模型结构
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.dense1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

# 将 TensorFlow 模型的参数转移到 PyTorch 模型中
pytorch_model = PyTorchModel()
with torch.no_grad():
    pytorch_model.dense1.weight = nn.Parameter(torch.tensor(tf_model.layers[0].get_weights()[0].T))
    pytorch_model.dense1.bias = nn.Parameter(torch.tensor(tf_model.layers[0].get_weights()[1]))
    pytorch_model.dense2.weight = nn.Parameter(torch.tensor(tf_model.layers[1].get_weights()[0].T))
    pytorch_model.dense2.bias = nn.Parameter(torch.tensor(tf_model.layers[1].get_weights()[1]))

# 3. 模型保存
# 保存转换后的 PyTorch 模型
torch.save(pytorch_model.state_dict(), 'pytorch_model.pth')
```

当然，实际操作远比这复杂，需要处理不同框架算子的细微差异、参数的命名规则、以及张量格式的适配问题，比如NCHW和NHWC的转换，可能还需要进行一些优化。

幸运的是，我们不需要每次都从零开始写转换器。现在已经有很多成熟的工具可以帮我们做这件事。这里列出部分可实现[不同框架迁移的模型转换器](https://chenzomi12.github.io/04Inference04Converter/04Detail.html#id5)

我们重点聊聊规范式转换，特别是ONNX。ONNX，全称Open Neural Network Exchange，是一个开放的、标准化的模型格式。它的目标是打破不同AI框架之间的壁垒，让模型可以在不同框架间自由流通。ONNX由微软、亚马逊、Meta这些大公司共同开发，得到了很多主流框架的支持，比如PyTorch、MXNet、Caffe2、TensorFlow等等。有了ONNX，开发者就可以在开发阶段选择最合适的框架，然后在部署时，再用ONNX导出模型，导入到目标框架或推理引擎中运行，非常灵活。

ONNX本身是什么样的呢？它本质上是一种可扩展的计算图模型。它定义了一系列标准的运算单元（OP）和数据类型。一个ONNX模型就是一个有向无环图，图中的每个节点就是一个算子，节点之间通过边连接，表示数据的流向。这种通用的计算图表示，使得不同框架构建的模型，只要遵循ONNX的规范，就可以被转换成ONNX格式，从而实现跨框架的互操作性。

我们来看一个PyTorch模型转ONNX的实例。

```py
x = torch.randn(1, 784)

# 导出为 ONNX 格式
with torch.no_grad():
    torch.onnx.export(
        pytorch_model,
        x,
        "pytorch_model.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    
import onnx 
 
onnx_model = onnx.load("pytorch_model.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")
```

首先，我们需要读取PyTorch模型，比如用torch.load加载。然后，我们需要提供一个示例输入，告诉ONNX导出器模型的输入格式。接着，调用torch.onnx.export函数，指定模型、输入、输出文件名、ONNX版本号、输入输出名称等参数。这个函数会把PyTorch模型转换成ONNX文件。最后，我们可以用onnx.checker.check_model来验证一下导出的ONNX文件是否格式正确。

导出的ONNX文件长什么样呢？我们可以用像Netron这样的工具来可视化。

<img src="https://chenzomi12.github.io/_images/04Detail02.png" alt="ONNX 模型可视化" style="zoom: 67%;" />

打开ONNX文件，你会看到输入、输出、以及中间的算子节点。点击输入或输出，可以看到它们的基本信息，比如名称、数据类型、维度。点击某个算子节点，比如Gemm，可以看到这个算子的详细信息，包括它的属性、输入输出、以及权重。这样就能直观地了解模型的结构和计算流程。这对于调试模型和理解模型转换结果非常有帮助。