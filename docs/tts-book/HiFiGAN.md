## HiFiGAN

### HiFiGAN概述

HiFiGAN是近年来在学术界和工业界都较为常用的声码器，能够将声学模型产生的频谱转换为高质量的音频，这种声码器采用生成对抗网络（Generative
Adversial
Networks，GAN）作为基础生成模型，相比于之前相近的MelGAN，改进点在于：

1.  引入了多周期判别器（Multi-Period
    Discriminator，MPD）。HiFiGAN同时拥有多尺度判别器（Multi-Scale
    Discriminator，MSD）和多周期判别器，尽可能增强GAN判别器甄别合成或真实音频的能力，从而提升合成音质。

2.  生成器中提出了多感受野融合模块。WaveNet为了增大感受野，叠加带洞卷积，音质虽然很好，但是也使得模型较大，推理速度较慢。HiFiGAN则提出了一种残差结构，交替使用带洞卷积和普通卷积增大感受野，保证合成音质的同时，提高推理速度。

### HiFiGAN生成器简介

HiFiGAN的生成器主要有两块，一个是上采样结构，具体由一维转置卷积组成；二是所谓的多感受野融合（Multi-Receptive
Field
Fusion，MRF）模块，主要负责对上采样获得的采样点进行优化，具体由残差网络组成。

### 上采样结构

作为声码器的生成器，不但需要负责将频谱从频域转换到时域，而且要进行上采样（upsampling）。以80维梅尔频谱合成16kHz的语音为例，假设帧移为10ms，则每个帧移内有160个语音样本点，需要通过80个梅尔频谱值获得，因此，需要利用卷积网络不断增加输出"长度"，降低输出"通道数"，直到上采样倍数达到160，通道数降低为1即可。

对于上采样操作，可以使用插值算法进行处理，比如最近邻插值（Nearest
neighbor interpolation）、双线性插值（Bi-Linear
interpolation）、双立方插值（Bi-Cubic
interpolation）等，但是这些插值算法说到底是人工规则，而神经网络可以自动学习合适的变换，[转置卷积（ConvTransposed）](https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose1d.html)，也称反卷积Deconvolution、微步卷积Fractionally-strided
Convolution，则是合适的上采样结构。一般的卷积中，每次卷积操作都是对输入张量和卷积核的每个元素进行相乘再加和，卷积的输入和输出是`多对一`的映射关系，而转置卷积则反过来，是`一对多`的映射关系。从计算机的内部实现来看，定义：

1.  $X$ 为输入张量，大小为 $X_{width}\times X_{height}$

2.  $Y$ 为输出张量，大小为 $Y_{width}\times Y_{height}$

3.  $C$ 为卷积核，大小为 $C_{width}\times C_{height}$

经过普通的卷积运算之后，将大张量 $X$ "下采样"到小张量 $Y$
。具体来说，首先将输入张量展平为向量，也即是
$[X_{width}\times X_{height},1]$ ，同时也将卷积核展平成向量到输入张量
$X$
的大小：由于卷积核小于输入张量，在行和列上都用0填充至输入张量大小，然后展平，则卷积核向量大小为
$[1,X_{width}\times X_{height}]$
；同时按照步长，左侧填充0偏移该卷积核向量，最终，卷积核向量的个数为输出张量元素个数，则构成的卷积核张量大小为
$[Y_{width}\times Y_{height},X_{width}\times X_{height}]$
，卷积核张量和输入张量矩阵乘，获得输出张量
$[Y_{width}\times Y_{height},1]$ ，重塑大小为 $C_{width},C_{height}$ 。

此时，如果使用卷积核张量的转置
$[X_{width}\times X_{height},Y_{width}\times Y_{height}]$ 矩阵乘展平的
$[Y_{width}\times Y_{height},1]$ ，得到的结果就是
$[X_{width}\times X_{height},1]$
，和刚刚的输入张量大小相同，这就完成了一次`转置卷积`。但实际上，上述操作并非可逆关系，卷积将输入张量"下采样"到输出张量，本质是有损压缩的过程，由于在卷积中使用的`卷积核张量`并非可逆矩阵，转置卷积操作之后并不能恢复到原始的数值，仅仅是恢复到原始的形状。这其实也就是线性谱与梅尔频谱关系，加权求和得到梅尔频谱之后就回不来了，顶多求梅尔滤波器组的伪逆，近似恢复到线性谱。

此外，在使用转置卷积时需要注意棋盘效应（Checkboard
artifacts）。棋盘效应主要是由于转置卷积的"不均匀重叠"（Uneven
overlap）造成的，输出中每个像素接受的信息量与相邻像素不同，在输出上找不到连续且均匀重叠的区域，表现是图像中一些色块的颜色比周围色块要深，像棋盘上的方格，参见[Deconvolution
and Checkerboard
Artifacts](https://distill.pub/2016/deconv-checkerboard)。避免棋盘效应的方法主要有：kernel_size的大小尽可能被stride整除，尽可能使用stride=1的转置卷积；堆叠转置卷积减轻重叠；网络末尾使用
$1\times 1$ 的转置卷积等。

通过上述的原理部分，可以看出卷积和转置卷积是对偶运算，输入变输出，输出变输入，卷积的输入输出大小关系为：

$$L_{out}=\frac{L_{in}+2\times padding-kernel\_size}{stride}+1$$

那么转置卷积的输入输出大小则为：

$$L_{out}=(L_{in}-1)\times stride+kernel\_size-2\times padding$$

当然，加入dilation之后，大小计算稍复杂些，参见[Pytorch-ConvTranspose1d](https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose1d.html)，[Pytorch-Conv1d](https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html)。

该部分参考文献：

1.  [怎样通俗易懂地解释反卷积？](https://www.zhihu.com/question/48279880/answer/1682194600)

2.  [一文搞懂反卷积，转置卷积](https://zhuanlan.zhihu.com/p/158933003)

3.  [Deconvolution and Checkerboard
    Artifacts](https://distill.pub/2016/deconv-checkerboard)

4.  [如何去除生成图片产生的棋盘伪影？](https://www.zhihu.com/question/436832427/answer/1679396968)

5.  [A guide to convolution arithmetic for deep
    learning](https://arxiv.org/abs/1603.07285)

6.  [Pytorch-ConvTranspose1d](https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose1d.html)

7.  [Pytorch-Conv1d](https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html)

转置卷积实现的上采样层定义为：

    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
        self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel//(2**i), 
        h.upsample_initial_channel//(2**(i+1)),kernel_size=k, 
        stride=u, padding=(k-u)//2)))

对于hop_size=256来说，h.upsample_rates和h.upsample_kernel_sizes分别为：

1.  \"upsample_rates\": \[8,8,2,2\],

2.  \"upsample_kernel_sizes\": \[16,16,4,4\],

根据转置卷积的输入输出大小关系：

$$L_{out}=(L_{in}-1)\times stride-2\times padding+dilation\times (kernel\_size-1)+output\_padding+1$$

用于上采样的转置卷积，通过设置合适的padding，配合卷积核大小（kernel_size）和步进（stride），就可以实现输出与输入大小呈"步进倍数"的关系，在这里，卷积核（upsample_kernel_sizes）设置为步进（upsample_rates）的2倍。设置参数时，必须保持帧移点数，是各个卷积层步进（或者代码中所谓的上采样率update_rates）的乘积，在上例中，也就是：

$$hop\_length=256=8\times 8\times 2\times 2$$

### 多感受野融合

转置卷积的上采样容易导致棋盘效应，因此每次转置卷积上采样之后，都会跟着一个多感受野融合（MRF）的残差网络，以进一步提升样本点的生成质量。多感受野融合模块是一种利用带洞卷积和普通卷积提高生成器感受野的结构，带洞卷积的扩张倍数逐步递增，如dilation=1,3,5，每个带洞卷积之后，跟着卷积核大于1的普通卷积，从而实现带洞卷积和普通卷积的交替使用。带洞卷积和普通卷积的输入输出大小保持不变，在一轮带洞和普通卷积完成之后，原始输入跳连到卷积的结果，从而实现一轮"多感受野融合"。多感受野融合的具体实现上，论文中提出了两种参数量不同的残差网络。一种是参数量较多，多组带洞卷积（dilation=1,3,5）和普通卷积交替使用，HiFiGAN
v1 (config_v1.json)和HiFiGAN v2
(config_v2.json)均使用该种多感受野融合（MRF）模块。：

    class ResBlock1(torch.nn.Module):
        def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
            super(ResBlock1, self).__init__()
            self.h = h
            self.convs1 = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                  dilation=dilation[0],padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                  dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                  dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
            ])
            self.convs1.apply(init_weights)

            self.convs2 = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                    padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                    padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                    padding=get_padding(kernel_size, 1)))
            ])
            self.convs2.apply(init_weights)

        def forward(self, x):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c1(xt)
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                xt = c2(xt)
                x = xt + x
            return x

        def remove_weight_norm(self):
            for l in self.convs1:
                remove_weight_norm(l)
            for l in self.convs2:
                remove_weight_norm(l)

另外一种MRF大大减少了参数量，仅由两层带洞卷积（dilation=1,3）组成，但依然保持了跳跃连接的结构:

    class ResBlock2(torch.nn.Module):
        def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
            super(ResBlock2, self).__init__()
            self.h = h
            self.convs = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                  dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                  dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            ])
            self.convs.apply(init_weights)

        def forward(self, x):
            for c in self.convs:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c(xt)
                x = xt + x
            return x

        def remove_weight_norm(self):
            for l in self.convs:
                remove_weight_norm(l)

注意到两种MRF都使用了weight_norm对神经网络的权重进行规范化，相比于batch_norm，weight_norm不依赖mini-batch的数据，对噪音数据更为鲁棒；并且，可以应用于RNN等时序网络上；此外，weight_norm直接对神经网络的权重值进行规范化，前向和后向计算时，带来的额外计算和存储开销都较小。weight_norm本质是利用方向
$v$ 和幅度张量 $g$ 替代权重张量 $w$ ：

$$w=g\frac{v}{||v||}$$

方向张量 $v$ 和 $w$ 大小相同，幅度张量 $g$ 比 $w$ 少一维，使得 $w$
能够比较容易地整体缩放。不直接优化 $w$ ，而是训练 $v$ 和 $g$ 。

同时注意到，在推理时需要remove_weight_norm，这是因为训练时需要计算权重矩阵的方向和幅度张量，而在推理时，参数已经优化完成，要恢复回去，所以在推理时就直接移除weight_norm机制。

每个卷积核的0填充个数都调用了get_padding函数，利用填充保证输入输出的长宽大小一致，该填充大小的计算方法：

$$padding=(kernel\_size-1)*padding//2$$

### HiFiGAN判别器简介

HiFiGAN的判别器有两个，分别是多尺度和多周期判别器，从两个不同角度分别鉴定语音。多尺度判别器源自MelGAN声码器的做法，不断平均池化语音序列，逐次将语音序列的长度减半，然后在语音的不同尺度上施加若干层卷积，最后展平，作为多尺度判别器的输出。多周期判别器则是以不同的序列长度将一维的音频序列折叠为二维平面，在二维平面上施加二维卷积。

### 多尺度判别器

多尺度判别器的核心是多次平均池化，缩短序列长度，每次序列长度池化至原来的一半，然后进行卷积。具体来说，多尺度判别器首先对原样本点进行一次"原尺寸判别"，其中"原尺寸判别"模块中一维卷积的参数规范化方法为谱归一化（spectral_norm）；接着对样本点序列进行平均池化，依次将序列长度减半，然后对"下采样"的样本点序列进行判别，该模块中一维卷积的参数规范化方法为权重归一化（weight_norm）。在每一个特定尺度的子判别器中，首先进行若干层分组卷积，并对卷积的参数进行规范化；接着利用leaky_relu进行激活；在经过多个卷积层之后，最后利用输出通道为1的卷积层进行后处理，展平后作为输出。

    class MultiScaleDiscriminator(torch.nn.Module):
        def __init__(self):
            super(MultiScaleDiscriminator, self).__init__()
            self.discriminators = nn.ModuleList([
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ])
            self.meanpools = nn.ModuleList([
                AvgPool1d(4, 2, padding=2),
                AvgPool1d(4, 2, padding=2)
            ])

        def forward(self, y, y_hat):
            y_d_rs = []
            y_d_gs = []
            fmap_rs = []
            fmap_gs = []
            for i, d in enumerate(self.discriminators):
                if i != 0:
                    y = self.meanpools[i-1](y)
                    y_hat = self.meanpools[i-1](y_hat)
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
                y_d_rs.append(y_d_r)
                fmap_rs.append(fmap_r)
                y_d_gs.append(y_d_g)
                fmap_gs.append(fmap_g)

            return y_d_rs, y_d_gs, fmap_rs, fmap_gs

上述代码中y_d\_rs和y_d\_gs分别是真实和生成样本的多尺度判别器展平后的整体输出，fmap_rs和y_d\_gs分别是真实和生成样本经过每一层卷积的特征图（feature
map）。子判别器DiscriminatorS由若干层卷积组成，最后一层输出通道为1，之后对输出进行展平。注意到，与MelGAN不同，多尺度判别器的第一个子判别器DiscriminatorS使用谱归一化spectral_norm，之后两个子判别器则是正常使用权重归一化weight_norm规整可训练参数。谱归一化实际是在每次更新完可训练参数
$W$ 之后，都除以 $W$
的奇异值，以保证整个网络满足利普希茨连续性，使得GAN的训练更稳定。参见[GAN
的谱归一化(Spectral Norm)和矩阵的奇异值分解(Singular Value
Decompostion)](https://kaizhao.net/posts/spectral-norm)。DiscriminatorS的具体实现如下：

    class DiscriminatorS(torch.nn.Module):
        def __init__(self, use_spectral_norm=False):
            super(DiscriminatorS, self).__init__()
            norm_f = weight_norm if use_spectral_norm == False else spectral_norm
            self.convs = nn.ModuleList([
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ])
            self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

        def forward(self, x):
            fmap = []
            for l in self.convs:
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

            return x, fmap

x是子判别器展平后的整体输出，大小为\[B,l\]；fmap是经过卷积后的特征图（feature
map），类型为list，元素个数为卷积层数，上述代码中有8个卷积层，则fmap元素个数为8，每个元素均是大小为\[B,C,l'\]的张量。

### 多周期判别器

多周期判别器的重点是将一维样本点序列以一定周期折叠为二维平面，例如一维样本点序列\[1,2,3,4,5,6\]，如果以3为周期，折叠成二维平面则是\[\[1,2,3\],\[4,5,6\]\]，然后对这个二维平面施加二维卷积。具体来说，每个特定周期的子判别器首先进行填充，保证样本点数是周期的整倍数，以方便"折叠"为二维平面；接下来进入多个卷积层，输出通道数分别为\[32,128,512,1024\]，卷积之后利用leaky_relu激活，卷积层参数规范化方法均为权重归一化（weight_norm）；然后经过多个卷积层之后，利用一个输入通道数为1024，输出通道为1的卷积层进行后处理；最后展平，作为多周期判别器的最终输出。多周期判别器包含多个周期不同的子判别器，在论文代码中周期数分别设置为\[2,3,5,7,11\]。

    class MultiPeriodDiscriminator(torch.nn.Module):
        def __init__(self):
            super(MultiPeriodDiscriminator, self).__init__()
            self.discriminators = nn.ModuleList([
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ])

        def forward(self, y, y_hat):
            y_d_rs = []
            y_d_gs = []
            fmap_rs = []
            fmap_gs = []
            for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
                y_d_rs.append(y_d_r)
                fmap_rs.append(fmap_r)
                y_d_gs.append(y_d_g)
                fmap_gs.append(fmap_g)

            return y_d_rs, y_d_gs, fmap_rs, fmap_gs

上述代码中y_d\_rs和y_d\_gs分别是真实和生成样本的多周期判别器输出，fmap_rs和fmap_gs分别是真实和生成样本经过每一层卷积后输出的特征图（feature
map）。子判别器DiscriminatorP由若干层二维卷积组成：

    class DiscriminatorP(torch.nn.Module):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
            super(DiscriminatorP, self).__init__()
            self.period = period
            norm_f = weight_norm if use_spectral_norm == False else spectral_norm
            self.convs = nn.ModuleList([
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ])
            self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        def forward(self, x):
            fmap = []

            # 1d to 2d
            b, c, t = x.shape
            if t % self.period != 0: # pad first
                n_pad = self.period - (t % self.period)
                x = F.pad(x, (0, n_pad), "reflect")
                t = t + n_pad
            x = x.view(b, c, t // self.period, self.period)

            for l in self.convs:
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

            return x, fmap

x是子判别器展平后的整体输出，大小为\[B,l\]；fmap是经过每一层卷积后的特征图（feature
map），类型为list，元素个数为卷积层数，上述代码中有6个卷积层，则fmap元素个数为6，每个元素是大小为\[B,C,l',period\]的张量。

### 损失函数简介

HiFiGAN的损失函数主要包括三块，一个是GAN原始的生成对抗损失（GAN
Loss）；第二是梅尔频谱损失（Mel-Spectrogram
Loss），将生成音频转换回梅尔频谱之后，计算真实和生成音频对应梅尔频谱之间的L1距离；第三是特征匹配损失（Feature
Match Loss），主要是对比真实和合成音频在中间卷积层上的差异。

### 生成对抗损失

HiFiGAN仍然是一个生成对抗网络，判别器计算输入是真实样本的概率，生成器生成以假乱真的样本，最终达到生成器合成接近真实的样本，以致于判别器无法区分真实和生成样本。HiFiGAN使用[最小二乘GAN（LS-GAN）](https://zhuanlan.zhihu.com/p/25768099)，将原始GAN中的二元交叉熵替换为最小二乘损失函数。判别器的生成对抗损失定义为：

$${\rm L}_{Adv}(D;G)=\mathbb{E}_{(x,s)}[(D(x)-1)^2+(D(G(s)))^2]$$

对应的代码实现：

    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((dr-1)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

生成器的生成对抗损失定义为：

$${\rm L}_{Adv}(G;D)=\mathbb{E}_{s}[(D(G(s))-1)^2]$$

其中， $x$ 表示真实音频， $s$ 表示梅尔频谱。

对应的生成器代码实现：

    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((dg-1)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

更详尽关于GAN的理论参见：[GAN万字长文综述](https://zhuanlan.zhihu.com/p/58812258)

### 梅尔频谱损失

借鉴Parallel
WaveGAN等前人工作，向GAN中引入重建损失和梅尔频谱损失可以提高模型训练初期的稳定性、生成器的训练效率和合成语音的自然度。具体来说，梅尔频谱损失就是计算合成和真实语音对应频谱之间的L1距离：

$${\rm L}_{Mel}(G)=E_{(x,s)}[||\phi(x)-\phi(G(s))||_1]$$

其中， $\phi$ 表示将语音转换为梅尔频谱的映射函数。

对应的损失函数实现：

      loss_mel = F.l1_loss(y_mel, y_g_hat_mel)

上述代码中，y_mel表示真实语音对应的梅尔频谱，y_g\_hat_mel表示梅尔频谱合成语音之后，合成语音又转换回来得到的梅尔频谱。

### 特征匹配损失

特征匹配损失是用来度量神经网络从真实和合成语音中提取的特征差异，具体来说，就是计算真实和合成语音经过特征提取层之后输出之间的L1距离：

$${\rm L}_{FM}(G;D)=\mathbb{E}_{x,s}[\sum_{i=1}^T\frac{1}{N_i}||D^i(x)-D^i(G(s))||_1]$$

其中， $T$ 表示判别器中特征提取层的层数， $D^i$ 表示提取的特征， $N_i$
表示第 $i$ 层判别器网络提取的特征数量。对应的代码为：

    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

### 整体损失

1.  生成器的整体损失为：

    $${\rm L}_G={\rm L}_{Adv}(G;D)+\lambda_{fm}{\rm L}_{FM}(G;D)+\lambda_{mel}{\rm L}_{Mel}(G)$$

    其中， $\lambda_{fm}$ 和 $\lambda_{mel}$
    分别为特征匹配和梅尔频谱损失的加权系数，实验中
    $\lambda_{fm}=2,\lambda_{mel}=45$。

    因为HiFiGAN的判别器是由多尺度判别器和多周期判别器组成，因此生成器的总体损失又可以写作：

    $${\rm L}_G=\sum_{k=1}^K[{\rm L}_{Adv}(G;D_k)+\lambda_{fm}{\rm L}_{FM}(G;D_k)]+\lambda_{mel}{\rm L}_{Mel}(G)$$

    其中， $K$ 为多尺度判别器和多周期判别器的个数， $D_k$ 表示第 $k$
    个MPD和MSD的子判别器。

    对应的代码为：

          # L1 Mel-Spectrogram Loss
          loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
          
          y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
          y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
          loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
          loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
          loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
          loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
          loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

2.  判别器的整体损失为：

    $${\rm L}_D={\rm L}_{Adv}(D;G)$$

    类似于生成器，由于HiFiGAN拥有多个判别器，因此判别器的整体损失可以写作：

    $${\rm L}_D=\sum_{k=1}^K{\rm L}_{Adv}(D_k;G)$$

    其中， $K$ 为多尺度判别器和多周期判别器的个数， $D_k$ 表示第 $k$
    个MPD和MSD的子判别器。

    对应的代码为：

          # MPD
          y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
          loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

          # MSD
          y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
          loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

          loss_disc_all = loss_disc_s + loss_disc_f