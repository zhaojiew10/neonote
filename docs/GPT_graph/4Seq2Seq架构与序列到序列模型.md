今天深入探讨如何构建更强大的序列到序列模型，特别是Seq2Seq架构。序列到序列模型，顾名思义，它的核心任务就是将一个序列映射到另一个序列。这个序列可以是文本，也可以是其他符号序列。最早，人们尝试用一个单一的RNN来搞定整个序列到序列的任务，比如直接让RNN读完中文句子，然后输出英文句子。但很快发现，效果不理想。为什么呢？因为RNN在处理长序列时，就像一个记忆力有限的学者，读到后面可能就忘了前面的内容了。梯度消失、梯度爆炸问题，以及它捕捉长距离依赖的能力不足，使得信息在处理过程中容易丢失或混淆。

## 编码器-解码器架构

为了解决这个问题，研究人员提出了序列到序列模型的基石——编码器-解码器架构。这个想法非常巧妙：我们不再用一个RNN来处理所有事情，而是把它分成两个独立的部分。

- 一部分负责编码器，专门负责读取输入序列，比如中文，然后把它压缩成一个浓缩的精华，也就是一个固定大小的向量。
- 另一部分负责解码器，接收这个精华向量，然后像一个翻译官一样，根据这个向量，逐步生成输出序列，比如英文。这样分工合作，信息就更容易被完整地保存和传递了。

整个过程就像一个翻译机，先把原文翻译成一种通用语，再从通用语翻译成目标语言。

![image-20250504152951665](https://s2.loli.net/2025/12/02/5MYS37XfznDrg9E.png)

上图更具体地展示了基于RNN或LSTM的Seq2Seq模型。左边是输入序列，比如咖哥喜欢小雪，后面可能加了个EOS标记表示结束。右边是输出序列，比如KaGe likes XiaoXue，同样有EOS。注意**，输入序列的开头通常会加上一个特殊的SOS符号，表示序列开始**。**解码器在生成每个词的时候，会依赖于之前的词和当前的上下文状态，所以输出序列的开头也用SOS标记**。模型中间就是两个RNN或LSTM单元，分别负责编码和解码。

我们再细看一下这两个核心组件。

- 编码器，它的任务是吃进去一个完整的输入序列，比如中文句子，然后通过内部的RNN、LSTM或者GRU等机制，一步步处理，最终吐出一个浓缩的、固定大小的向量。这个向量，我们称之为**上下文向量，它包含了整个输入序列的所有信息。**
- 解码器，接收这个上下文向量作为起点，然后也开始一步步处理，生成一个输出序列。

![image-20250504153150286](https://s2.loli.net/2025/12/02/foCcrgh37jI8qxe.png)

左边是中文句子咖哥很喜欢小冰，经过编码器处理，变成了一个向量，比如02 85 03 12 99。这个向量就是编码器的输出。然后，这个向量被传递给右边的解码器，解码器根据这个向量，逐步生成英文句子KaGe very likes XiaoBing。注意，解码器的输出是逐个词生成的，直到遇到EOS结束标记。整个过程就是从输入序列到向量，再到输出序列的完整映射。

所以，我们可以这样总结Seq2Seq的核心思想：它本质上是一个压缩与解压缩的过程。输入序列被压缩成一个紧凑的向量，然后这个向量被解压缩成输出序列。这个压缩和解压缩的过程，通常由RNN、LSTM或GRU等序列建模方法来实现。

特别要注意的是，**解码器在生成序列时，它不是一次性生成所有词，而是一个词一个词地生成，而且当前生成的词，会作为下一个词的输入，这就是所谓的自回归特性**。这种特性使得模型能够更好地处理序列的生成任务。

Seq2Seq架构有几个非常重要的特点。

- 非常灵活，能够处理输入和输出序列长度不等的情况。比如，一个中文句子可能对应一个很长的英文句子，或者一个长句子对应一个短摘要。
- 它组件可以替换。编码器和解码器可以选用不同的RNN变体，比如LSTM、GRU，甚至可以引入更复杂的结构，比如Transformer。
- 具有很好的扩展性。我们可以在这个基础上，进一步添加注意力机制等组件，来提升模型在处理长序列和复杂语境时的能力。

## Seq2Seq模型实践

理论讲完了，我们来动手实践一下。我们来构建一个简单的Seq2Seq模型，让它能够学习如何把中文翻译成英文。

1. 构建实验语料库和词汇表
2. 生成Seq2Seq训练数据
3. 定义编码器和解码器类
4. 定义Seq2Seq架构
5. 训练Seq2Seq架构
6. 测试Seq2Seq架构

### 数据准备

数据准备。我们需要一个包含中文和英文句子对的语料库。

```py
sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']]
```

有了语料库，我们还需要构建词汇表。我们需要知道每个词对应的编号。我们分别创建一个中文词汇表和一个英文词汇表，把所有出现的词都收集起来。然后，为每个词分配一个唯一的编号，也就是索引。同时，我们还要创建一个反向索引，知道每个编号对应哪个词。这样，模型就能把文本转换成数字序列，方便神经网络处理。

```py
中文词汇到索引的字典： {'神经网络': 0, '小冰': 1, '人工智能': 2, '复杂': 3, '我': 4, '处理': 5, '改变': 6, '深度学习': 7, '强大': 8, '咖哥': 9, '世界': 10, '学习': 11, '自然': 12, '语言': 13, '很': 14, '非常': 15, '爱': 16, '喜欢': 17}

英文词汇到索引的字典： {'are': 0, 'the': 1, 'NLP': 2, 'XiaoBing': 3, 'I': 4, 'so': 5, 'complex': 6, '<sos>': 7, 'studying': 8, 'KaGe': 9, 'DL': 10, 'AI': 11, 'love': 12, 'is': 13, 'world': 14, 'Neural-Nets': 15, 'changed': 16, 'powerful': 17, '<eos>': 18, 'likes': 19}
```

### 生成Seq2Seq训练数据

我们需要把刚才的语料库转换成模型可以直接使用的训练数据。具体来说，就是把每个句子的中文单词、带sos的英文单词和带eos的英文单词都转换成对应的索引序列。然后，把这些索引序列转换成PyTorch的LongTensor张量格式。

```py
原始句子： ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>']
编码器输入张量的形状： torch.Size([1, 4])
解码器输入张量的形状： torch.Size([1, 5])
目标张量的形状： torch.Size([1, 5])

编码器输入张量： tensor([[ 5, 13, 10,  4]]) # 每个词查索引表 
解码器输入张量： tensor([[17, 14, 12, 18,  9]])
目标张量： tensor([[14, 12, 18,  9, 16]])
```

这个函数每次调用，都会返回一个批次的训练数据。这个函数的输出结果是这样的：`encoder_input` 是中文句子的索引张量，`decoder_input` 是带 `<sos>` 的英文句子的索引张量，`target` 是带 `<eos>` 的英文句子的索引张量。注意 `decoder_input` 和 `target` 的区别在于 `<sos>` 和 `<eos>` 的位置。`decoder_input` 用于训练解码器，而 `target` 用于计算损失。这个 `decoder_input` 里面包含了真实的目标序列，这其实是教师强制的体现。

这里提到了一个重要的概念：**教师强制**。简单来说，就是在训练解码器的时候，我们给它喂的是真实答案。具体来说，就是把目标序列的英文单词，除了最后一个 `<eos>`，都作为输入给解码器。这样做是**为了让模型更快地学习到正确的翻译路径**。但是，这也会带来一个问题：模型在训练时依赖了真实答案，但在实际应用中，它只能自己生成下一个词，这就可能造成训练和测试时的分布不一致，也就是所谓的曝光偏差。为了解决这个问题，可以使用计划采样，让模型在训练过程中逐渐适应自己生成的词。

### 定义编码器和解码器类

我们开始构建神经网络模型了。

```python
import torch.nn as nn # 导入 torch.nn 库
# 定义编码器类，继承自 nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小       
        self.embedding = nn.Embedding(input_size, hidden_size) # 创建词嵌入层       
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # 创建 RNN 层    
    def forward(self, inputs, hidden): # 前向传播函数
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量       
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入 RNN 层并获取输出
        return output, hidden
    
# 定义解码器类，继承自 nn.Module
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小       
        self.embedding = nn.Embedding(output_size, hidden_size) # 创建词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # 创建 RNN 层       
        self.out = nn.Linear(hidden_size, output_size) # 创建线性输出层    
    def forward(self, inputs, hidden):  # 前向传播函数     
        # inputs = tensor([[ 7,  2, 13,  5, 17]])
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量       
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入 RNN 层并获取输出       
        output = self.out(output) # 使用线性层生成最终输出
        return output, hidden

n_hidden = 128 # 设置隐藏层数量
# 创建编码器和解码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)

编码器结构： Encoder(
  (embedding): Embedding(18, 128)
  (rnn): RNN(128, 128, batch_first=True)
)
 
解码器结构： Decoder(
  (embedding): Embedding(20, 128)
  (rnn): RNN(128, 128, batch_first=True)
  (out): Linear(in_features=128, out_features=20, bias=True)
)
```

我们需要定义两个类：Encoder和Decoder。这两个类都继承自PyTorch的nn.Module。它们的核心组件包括：

- 嵌入层，用来把输入的单词索引转换成低维的向量表示；
- RNN层，用来处理序列信息；
- **对于解码器，还需要一个线性输出层**，用来把RNN的输出转换成最终的词汇表概率分布。

这就是我们定义的编码器和解码器的代码。可以看到

- Encoder类主要负责处理输入序列，它包含一个嵌入层和一个RNN层
- Decoder类除了嵌入层和RNN层，还多了一个输出层，用于生成最终的预测结果。
- forward函数定义了数据如何在网络中流动。

这里解释一下RNN输出的两个值：**output 和 hidden**。

- output 是每个时间步的输出，可以理解为对当前输入的编码。
- hidden 是隐藏状态，它**包含了从序列开始到现在所有信息的累积，是RNN的核心记忆**。

在编码器中，hidden 状态会传递给解码器作为初始状态。**而编码器的 output，虽然在这个简单的模型里没直接用，但在后续的注意力机制中会发挥重要作用**。解码器的 output 则是我们最终需要的预测概率分布。

### 定义Seq2Seq架构

我们需要把编码器和解码器这两个组件组装起来，形成一个完整的Seq2Seq模型。

```py
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_input, hidden, dec_input):    # 定义前向传播函数
        # 使输入序列通过编码器并获取输出和隐藏状态
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)
        # 将编码器的隐藏状态传递给解码器作为初始隐藏状态
        decoder_hidden = encoder_hidden
        # 使解码器输入（目标序列）通过解码器并获取输出
        decoder_output, _ = self.decoder(dec_input, decoder_hidden)
        return decoder_output
# 创建 Seq2Seq 架构
model = Seq2Seq(encoder, decoder)

S2S 模型结构： Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(18, 128)
    (rnn): RNN(128, 128, batch_first=True)
  )
  (decoder): Decoder(
    (embedding): Embedding(20, 128)
    (rnn): RNN(128, 128, batch_first=True)
    (out): Linear(in_features=128, out_features=20, bias=True)
  )
)
```

我们定义一个名为Seq2Seq的类，它也继承自nn.Module。在这个类里，我们会初始化一个编码器和一个解码器对象。然后，定义一个forward函数，描述数据如何**从输入序列经过编码器，得到隐藏状态，再传递给解码器**，最终生成输出序列的过程。这就是我们定义的Seq2Seq模型类代码。

我们再来看一下这个forward函数的细节。它接收三个参数：

- enc_input，编码器的输入
- hidden，初始隐藏状态
- dec_input，解码器的输入序列

注意这个 dec_input，它包含了我们之前提到的带 `<sos>` 的真实目标序列。这再次体现了教师强制的策略。函数返回的是 decoder_output，它是一个张量，每个时间步对应一个词汇表的预测概率分布。

### 训练模型

我们定义一个训练函数，然后调用它来训练我们的Seq2Seq模型。

```py
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
       encoder_input, decoder_input, target = make_data(sentences) # 训练数据的创建
       # encoder_input = tensor([[ 9, 17,  1]])   
       # decoder_input = tensor([[ 7,  9, 19,  3]])   
       # target = tensor([[ 9, 19,  3, 18]])
       hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态      
       optimizer.zero_grad()# 梯度清零        
       output = model(encoder_input, hidden, decoder_input) # 获取模型输出        
       loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 计算损失        
       if (epoch + 1) % 40 == 0: # 打印损失
          print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")         
       loss.backward()# 反向传播        
       optimizer.step()# 更新参数
# 训练模型
epochs = 400 # 训练轮次
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器
train_seq2seq(model, criterion, optimizer, epochs) # 调用函数训练模型
```

可以看到，它是一个循环，每次迭代一个epoch。在每次迭代中，我们首先生成数据，初始化隐藏状态，然后执行前向传播、计算损失、反向传播、更新参数这几个步骤。我们还打印了每隔一定轮次的损失值，用来监控训练进度。

这里解释一下代码中的一些细节。我们**使用了nn.CrossEntropyLoss作为损失函数，因为它适合处理多分类问题**，比如预测下一个词。view函数用于改变张量的形状，比如把输出张量展平成二维，把目标标签展平成一维，以便损失函数计算。我们使用了Adam优化器，这是一个非常流行的优化算法。整个训练过程就是标准的深度学习流程。

### 测试模型

测试模型。我们定义一个测试函数，用来评估模型在新数据上的表现。

```py
# 定义测试函数
def test_seq2seq(model, source_sentence):
    # 将输入的句子转换为索引
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # [[12 13  5 14  8]]
    # 构建输出的句子的索引，以 '<sos>' 开始，后面跟 '<eos>'，长度与输入句子相同
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']]*(len(encoder_input[0])-1))
    # [ 7 18 18 18 18]
    # 转换为 LongTensor 类型
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0) # 增加一维    
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态    
    predict = model(encoder_input, hidden, decoder_input) # 获取模型输出    
    predict = predict.data.max(2, keepdim=True)[1] # 获取概率最大的索引
    # 打印输入的句子和预测的句子
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])
# 测试模型
test_seq2seq(model, '自然 语言 处理 很 强大')
```

测试时，我们不再提供真实的英文句子作为解码器的输入，而是**只提供一个sos作为起始信号**。

然后，我们让模型自己一步步生成英文单词，直到遇到eos。我们把模型预测的索引转换回英文单词，看看翻译结果如何。这就是测试函数的代码。它首先将输入的中文句子转换为索引。然后，**它只构建了一个包含sos的解码器输入序列，后面填充了eos**（这个代码为了简化，没有实现真正的逐个token生成，而是直接输出了与输入序列等长的输出）。

接着，我们初始化隐藏状态，让模型通过前向传播，得到预测结果。我们取每个时间步概率最大的那个词作为输出，最后将这些索引转换回英文单词。可以看到，模型基本学到了翻译任务。不过，这里需要指出的是，**这个代码为了简化，没有实现真正的逐个token生成，而是直接输出了与输入序列等长的输出**。

对于更复杂的任务，我们需要实现更精细的生成过程，比如GPT模型那样，一个词一个词地生成。