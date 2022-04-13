## Tacotron

### Tacotron-2简介

以最常使用的Tacotron-2声学模型为例。原始论文参见：

1.  [Tacotron: Towards End-to-End Speech
    Synthesis](https://arxiv.org/abs/1703.10135)

2.  [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
    Predictions](https://arxiv.org/abs/1712.05884)

此外，谷歌在语音合成领域，特别是端到端语音合成领域做出了开创性的共享，该组会将最新的论文汇总在[Tacotron
(/täkōˌträn/): An end-to-end speech synthesis system by
Google](https://google.github.io/tacotron/).

![Tacotron-2模型结构 ](../asset/tacotron2_arch.png)

### 声学特征建模网络

Tacotron-2的声学模型部分采用典型的序列到序列结构。编码器是3个卷积层和一个双向LSTM层组成的模块，卷积层给予了模型类似于N-gram感知上下文的能力，并且对不发音字符更加鲁棒。经词嵌入的注音序列首先进入卷积层提取上下文信息，然后送入双向LSTM生成编码器隐状态。编码器隐状态生成后，就会被送入注意力机制，以生成编码向量。我们利用了一种被称为位置敏感注意力（Location
Sensitive Attention，LSA），该注意力机制的对齐函数为：

$$score(s_{i-1},h_j)=v_a^T{\rm tanh}(Ws_{i-1}+Vh_j+Uf_{i,j}+b)$$

其中， $v_a,W,V,U$为待训练参数， $b$ 是偏置值， $s_{i-1}$ 为上一时间步
$i-1$ 的解码器隐状态， $h_j$ 为当前时间步 $j$ 的编码器隐状态， $f_{i,j}$
为上一个解码步的注意力权重 $\alpha_{i-1}$ 经卷积获得的位置特征，如下式：

$$f_{i,j}=F*\alpha_{i-1}$$

其中， $\alpha_{i-1}$
是经过softmax的注意力权重的累加和。位置敏感注意力机制不但综合了内容方面的信息，而且关注了位置特征。解码过程从输入上一解码步或者真实音频的频谱进入解码器预处理网络开始，到线性映射输出该时间步上的频谱帧结束，模型的解码过程如下图所示。

![Tacotron2解码过程 ](../asset/tacotron2_decoder.png)

频谱生成网络的解码器将预处理网络的输出和注意力机制的编码向量做拼接，然后整体送入LSTM中，LSTM的输出用来计算新的编码向量，最后新计算出来的编码向量与LSTM输出做拼接，送入映射层以计算输出。输出有两种形式，一种是频谱帧，另一种是停止符的概率，后者是一个简单二分类问题，决定解码过程是否结束。为了能够有效加速计算，减小内存占用，引入缩减因子r（Reduction
Factor），即每一个时间步允许解码器预测r个频谱帧进行输出。解码完成后，送入后处理网络处理以生成最终的梅尔频谱，如下式所示。

$$s_{final}=s_i+s_i'$$

其中， $s_i$ 是解码器输出， $s_{final}$ 表示最终输出的梅尔频谱， $s_i'$
是后处理网络的输出，解码器的输出经过后处理网络之后获得 $s_i'$ 。

在Tacotron-2原始论文中，直接将梅尔频谱送入声码器WaveNet生成最终的时域波形。但是WaveNet计算复杂度过高，几乎无法实际使用，因此可以使用其它声码器，比如Griffin-Lim、HiFiGAN等。

### 损失函数

Tacotron2的损失函数主要包括以下4个方面：

1.  进入后处理网络前后的平方损失。

    $${\rm MelLoss}=\frac{1}{n}\sum_{i=1}^n(y_{real,i}^{mel}-y_{before,i}^{mel})^2+\frac{1}{n}\sum_{i=1}^n(y_{real,i}^{mel}-y_{after,i}^{mel})^2$$

    其中， $y_{real,i}^{mel}$ 表示从音频中提取的真实频谱，
    $y_{before,i}^{mel},y_{after,i}^{mel}$
    分别为进入后处理网络前、后的解码器输出， $n$ 为每批的样本数。

2.  从CBHG模块中输出线性谱的平方损失。

    $${\rm LinearLoss}=\frac{1}{n}\sum_{i=1}^{n}(y_{real,i}^{linear}-y_{i}^{linear})^2$$

    其中， $y_{real,i}^{linear}$ 是从真实语音中计算获得的线性谱，
    $y_{i}^{linear}$ 是从CBHG模块输出的线性谱。

3.  停止符交叉熵

    $${\rm StopTokenLoss}=-[y\cdot {\rm log}(p)+(1-y)\cdot {\rm log}(1-p)]$$

    其中， $y$ 为停止符真实概率分布， $p$
    是解码器线性映射输出的预测分布。

4.  L2正则化 $${\rm RegulationLoss}=\frac{1}{K}\sum_{k=1}^K w_k^2$$
    其中， $K$ 为参数总数， $w_k$
    为模型中的参数，这里排除偏置值、RNN以及线性映射中的参数。最终的损失函数为上述4个部分的损失之和，如下式：

    $${\rm Loss}={\rm MelLoss}+{\rm LinearLoss}+{\rm StopTokenLoss}+{\rm RegulationLoss}$$