## Griffin-Lim声码器

在早期的很多Tacotron开源语音合成模型中均采用Griffin-Lim声码器，同时也有一些专门的开源实现，比如[GriffinLim](https://github.com/bkvogel/griffin_lim)。

### 算法原理

原始的音频很难提取特征，需要进行傅里叶变换将时域信号转换到频域进行分析。音频进行傅里叶变换后，结果为复数，复数的绝对值就是幅度谱，而复数的实部与虚部之间形成的角度就是相位谱。经过傅里叶变换之后获得的幅度谱特征明显，可以清楚看到基频和对应的谐波。基频一般是声带的频率，而谐波则是声音经过声道、口腔、鼻腔等器官后产生的共振频率，且频率是基频的整数倍。

Griffin-Lim将幅度谱恢复为原始波形，但是相比原始波形，幅度谱缺失了原始相位谱信息。音频一般采用的是短时傅里叶变化，因此需要将音频分割成帧（每帧20ms 50ms），再进行傅里叶变换，帧与帧之间是有重叠的。Griffin-Lim算法利用两帧之间有重叠部分的这个约束重构信号，因此如果使用Griffin-Lim算法还原音频信号，就需要尽量保证两帧之间重叠越多越好，一般帧移为每一帧长度的25%左右，也就是帧之间重叠75%为宜。

Griffin-Lim在已知幅度谱，不知道相位谱的情况下重建语音，算法的实现较为简单，整体是一种迭代算法，迭代过程如下：

1.  随机初始化一个相位谱；

2.  用相位谱和已知的幅度谱经过逆短时傅里叶变换（ISTFT）合成新语音；

3.  对合成的语音做短时傅里叶变换，得到新的幅度谱和相位谱；

4.  丢弃新的幅度谱，用相位谱和已知的幅度谱合成语音，如此重复，直至达到设定的迭代轮数。

在迭代过程中，预测序列与真实序列幅度谱之间的距离在不断缩小，类似于EM算法。推导过程参见：[Griffin
Lim算法的过程和证明](https://zhuanlan.zhihu.com/p/102539783)和[Griffin
Lim声码器介绍](https://zhuanlan.zhihu.com/p/66809424)。

### 代码实现

摘抄自[Build End-To-End TTS Tacotron: Griffin Lim
信号估计算法](https://zhuanlan.zhihu.com/p/25002923)。

    def griffin_lim(stftm_matrix, shape, min_iter=20, max_iter=50, delta=20):
      y = np.random.random(shape)
      y_iter = []

      for i in range(max_iter):
          if i >= min_iter and (i - min_iter) % delta == 0:
              y_iter.append((y, i))
          stft_matrix = librosa.core.stft(y)
          stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
          y = librosa.core.istft(stft_matrix)
      y_iter.append((y, max_iter))

      return y_iter

具体使用：

    # assume 1 channel wav file
    sr, data = scipy.io.wavfile.read(input_wav_path)

    # 由 STFT -> STFT magnitude
    stftm_matrix = np.abs(librosa.core.stft(data))
    # + random 模拟 modification
    stftm_matrix_modified = stftm_matrix + np.random.random(stftm_matrix.shape)

    # Griffin-Lim 估计音频信号
    y_iters = griffin_lim(stftm_matrix_modified, data.shape)