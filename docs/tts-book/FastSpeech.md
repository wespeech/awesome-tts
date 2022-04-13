## FastSpeech

FastSpeech是基于Transformer显式时长建模的声学模型，由微软和浙大提出。原始论文参见：

1.  [FastSpeech: Fast, Robust and Controllable Text to
    Speech](https://arxiv.org/abs/1905.09263)

2.  [FastSpeech 2: Fast and High-Quality End-to-End Text to
    Speech](https://arxiv.org/abs/2006.04558)

相对应地，微软在语音合成领域的论文常常发布在[Microsoft-Speech
Research](https://speechresearch.github.io/)。

![FastSpeech 2整体结构](../asset/fastspeech_arch_paper.png)

### 模型结构

FastSpeech 2和上代FastSpeech的编解码器均是采用FFT（feed-forward
Transformer，前馈Transformer）块。编解码器的输入首先进行位置编码，之后进入FFT块。FFT块主要包括多头注意力模块和位置前馈网络，位置前馈网络可以由若干层Conv1d、LayerNorm和Dropout组成。

论文中提到语音合成是典型的一对多问题，同样的文本可以合成无数种语音。上一代FastSpeech主要通过目标侧使用教师模型的合成频谱而非真实频谱，以简化数据偏差，减少语音中的多样性，从而降低训练难度；向模型提供额外的时长信息两个途径解决一对多的问题。在语音中，音素时长自不必说，直接影响发音长度和整体韵律；音调则是影响情感和韵律的另一个特征；能量则影响频谱的幅度，直接影响音频的音量。在FastSpeech
2中对这三个最重要的语音属性单独建模，从而缓解一对多带来的模型学习目标不确定的问题。

在对时长、基频和能量单独建模时，所使用的网络结构实际是相似的，在论文中称这种语音属性建模网络为变量适配器（Variance
Adaptor）。时长预测的输出也作为基频和能量预测的输入。最后，基频预测和能量预测的输出，以及依靠时长信息展开的编码器输入元素加起来，作为下游网络的输入。变量适配器主要是由2层卷积和1层线性映射层组成，每层卷积后加ReLU激活、LayerNorm和Dropout。代码摘抄自[FastSpeech2](https://github.com/ming024/FastSpeech2)，添加了一些注释。

    class VariancePredictor(nn.Module):
      """ Duration, Pitch and Energy Predictor """
      def __init__(self):
          super(VariancePredictor, self).__init__()

          self.input_size = hp.encoder_hidden
          self.filter_size = hp.variance_predictor_filter_size
          self.kernel = hp.variance_predictor_kernel_size
          self.conv_output_size = hp.variance_predictor_filter_size
          self.dropout = hp.variance_predictor_dropout

          self.conv_layer = nn.Sequential(OrderedDict([
              ("conv1d_1", Conv(self.input_size,
                                self.filter_size,
                                kernel_size=self.kernel,
                                padding=(self.kernel-1)//2)),
              ("relu_1", nn.ReLU()),
              ("layer_norm_1", nn.LayerNorm(self.filter_size)),
              ("dropout_1", nn.Dropout(self.dropout)),
              ("conv1d_2", Conv(self.filter_size,
                                self.filter_size,
                                kernel_size=self.kernel,
                                padding=1)),
              ("relu_2", nn.ReLU()),
              ("layer_norm_2", nn.LayerNorm(self.filter_size)),
              ("dropout_2", nn.Dropout(self.dropout))
          ]))

          self.linear_layer = nn.Linear(self.conv_output_size, 1)

      def forward(self, encoder_output, mask):
          '''
          :param encoder_output: Output of encoder. [batch_size,seq_len,encoder_hidden]
          :param mask: Mask for encoder. [batch_size,seq_len]
          '''
          out = self.conv_layer(encoder_output)
          out = self.linear_layer(out)
          out = out.squeeze(-1)

          if mask is not None:
              out = out.masked_fill(mask, 0.)

          return out

利用该变量适配器对时长、基频和能量进行建模。

    class VarianceAdaptor(nn.Module):
      """ Variance Adaptor """

      def __init__(self):
          super(VarianceAdaptor, self).__init__()
          self.duration_predictor = VariancePredictor()
          self.length_regulator = LengthRegulator()
          self.pitch_predictor = VariancePredictor()
          self.energy_predictor = VariancePredictor()

          self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(
              np.log(hp.f0_min), np.log(hp.f0_max), hp.n_bins-1)), requires_grad=False)
          self.energy_bins = nn.Parameter(torch.linspace(
              hp.energy_min, hp.energy_max, hp.n_bins-1), requires_grad=False)
          self.pitch_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
          self.energy_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)

      def forward(self, x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None, d_control=1.0, p_control=1.0, e_control=1.0):
          '''
          :param x: Output of encoder. [batch_size,seq_len,encoder_hidden]
          :param src_mask: Mask of encoder, can get src_mask form input_lengths. [batch_size,seq_len]
          :param duration_target, pitch_target, energy_target: Ground-truth when training, None when synthesis. [batch_size,seq_len]
          '''
          log_duration_prediction = self.duration_predictor(x, src_mask)
          if duration_target is not None:
              x, mel_len = self.length_regulator(x, duration_target, max_len)
          else:
              duration_rounded = torch.clamp(
                  (torch.round(torch.exp(log_duration_prediction)-hp.log_offset)*d_control), min=0)
              x, mel_len = self.length_regulator(x, duration_rounded, max_len)
              mel_mask = utils.get_mask_from_lengths(mel_len)

          pitch_prediction = self.pitch_predictor(x, mel_mask)
          if pitch_target is not None:
              pitch_embedding = self.pitch_embedding(
                  torch.bucketize(pitch_target, self.pitch_bins))
          else:
              pitch_prediction = pitch_prediction*p_control
              pitch_embedding = self.pitch_embedding(
                  torch.bucketize(pitch_prediction, self.pitch_bins))

          energy_prediction = self.energy_predictor(x, mel_mask)
          if energy_target is not None:
              energy_embedding = self.energy_embedding(
                  torch.bucketize(energy_target, self.energy_bins))
          else:
              energy_prediction = energy_prediction*e_control
              energy_embedding = self.energy_embedding(
                  torch.bucketize(energy_prediction, self.energy_bins))

          x = x + pitch_embedding + energy_embedding

          return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask

同样是通过长度调节器（Length
Regulator），利用时长信息将编码器输出长度扩展到频谱长度。具体实现就是根据duration的具体值，直接上采样。一个音素时长为2，就将编码器输出复制2份，给3就直接复制3份，拼接之后作为最终的输出。实现代码：

    class LengthRegulator(nn.Module):
      """ Length Regulator """

      def __init__(self):
          super(LengthRegulator, self).__init__()

      def LR(self, x, duration, max_len):
          '''
          :param x: Output of encoder. [batch_size,phoneme_seq_len,encoder_hidden]
          :param duration: Duration for phonemes. [batch_size,phoneme_seq_len]
          :param max_len: Max length for mel-frames. scaler

          Return:
          output: Expanded output of encoder. [batch_size,mel_len,encoder_hidden]
          '''
          output = list()
          mel_len = list()
          for batch, expand_target in zip(x, duration):
              # batch: [seq_len,encoder_hidden]
              # expand_target: [seq_len]
              expanded = self.expand(batch, expand_target)
              output.append(expanded)
              mel_len.append(expanded.shape[0])

          if max_len is not None:
              output = utils.pad(output, max_len)
          else:
              output = utils.pad(output)

          return output, torch.LongTensor(mel_len).to(device)

      def expand(self, batch, predicted):
          out = list()

          for i, vec in enumerate(batch):
              # expand_size: scaler
              expand_size = predicted[i].item()
              # Passing -1 as the size for a dimension means not changing the size of that dimension.
              out.append(vec.expand(int(expand_size), -1))
          out = torch.cat(out, 0)

          return out

      def forward(self, x, duration, max_len):
          output, mel_len = self.LR(x, duration, max_len)
          return output, mel_len

对于音高和能量的预测，模块的主干网络相似，但使用方法有所不同。以音高为例，能量的使用方式相似。首先对预测出的实数域音高值进行分桶，映射为一定范围内的自然数集，然后做嵌入。

    pitch_prediction = self.pitch_predictor(x, mel_mask)
    if pitch_target is not None:
      pitch_embedding = self.pitch_embedding(
          torch.bucketize(pitch_target, self.pitch_bins))
    else:
      pitch_prediction = pitch_prediction*p_control
      pitch_embedding = self.pitch_embedding(
          torch.bucketize(pitch_prediction, self.pitch_bins))

这里用到了Pytorch中一个不是特别常见的函数[torch.bucketize](https://pytorch.org/docs/master/generated/torch.bucketize.html)。这是Pytorch中的分桶函数，boundaries确定了各个桶的边界，是一个单调递增向量，用于划分input，并返回input所属桶的索引，桶索引从0开始。

能量嵌入向量的计算方法与之类似。至此，获得了展开之后的编码器输出x，基频嵌入向量`pitch_embedding`和能量嵌入向量`energy_embedding`之后，元素加获得最终编解码器的输入。

### 损失函数

FastSpeech
2的目标函数由PostNet前后的频谱均方差，时长、音高和能量的均方差组成。时长映射到指数域（时长预测器输出的数值
$x$ 作为指数，最终的预测时长为 $e^x$
），音高映射到对数域（音高预测器输出的数值 $x$ 做对数，作为最终的音高
${\rm log} x$ ），而能量直接采用能量预测器的输出值。整体的损失函数为：

$${\rm Loss}={\rm Loss}_{mel}+{\rm Loss}_{mel}^{post}+{\rm Loss}_{duration}+{\rm Loss}_{pitch}+{\rm Loss}_{energy}$$

频谱的损失函数形式采用均方差（MSE），时长、基频和能量采用平均绝对误差（MAE），具体的实现如下：

    log_d_target.requires_grad = False
    p_target.requires_grad = False
    e_target.requires_grad = False
    mel_target.requires_grad = False

    log_d_predicted = log_d_predicted.masked_select(src_mask)
    log_d_target = log_d_target.masked_select(src_mask)
    p_predicted = p_predicted.masked_select(mel_mask)
    p_target = p_target.masked_select(mel_mask)
    e_predicted = e_predicted.masked_select(mel_mask)
    e_target = e_target.masked_select(mel_mask)

    mel = mel.masked_select(mel_mask.unsqueeze(-1))
    mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
    mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

    mel_loss = self.mse_loss(mel, mel_target)
    mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

    d_loss = self.mae_loss(log_d_predicted, log_d_target)
    p_loss = self.mae_loss(p_predicted, p_target)
    e_loss = self.mae_loss(e_predicted, e_target)

    total_loss = mel_loss + mel_postnet_loss + d_loss + p_loss + e_loss

### 小结

FastSpeech系列的声学模型将Transformer引入语音合成领域，并且显式建模语音中的重要特征，比如时长、音高和能量等。实际上，微软首次在[Neural
Speech Synthesis with Transformer
Network](https://arxiv.org/abs/1809.08895)将Transformer作为主干网络，实现语音合成的声学模型，这一思想同样被[FastPitch:
Parallel Text-to-speech with Pitch
Prediction](https://arxiv.org/abs/2006.06873)采用，相关的开源代码：[as-ideas/TransformerTTS](https://github.com/as-ideas/TransformerTTS)。