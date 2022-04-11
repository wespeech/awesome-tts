## STRAIGHT声码器

### 概述

STARIGHT（Speech Transformation and Representation using Adaptive
Interpolation of weiGHTed
spectrum），即利用自适应加权谱内插进行语音转换和表征。STRAIGHT将语音信号解析成相互独立的频谱参数（谱包络）和基频参数（激励部分），能够对语音信号的基频、时长、增益、语速等参数进行灵活的调整，该模型在分析阶段仅针对语音基音、平滑功率谱和非周期成分3个声学参数进行分析提取，在合成阶段利用上述3个声学参数进行语音重构。

STRAIGHT采用源-滤波器表征语音信号，可将语音信号看作激励信号通过时变线性滤波器的结果。

对于能量信号和周期信号，其傅里叶变换收敛，因此可以用频谱（Spectrum）来描述；对于随机信号，傅里叶变换不收敛，因此不能用频谱进行描述，而应当使用功率谱（PSD），不严谨地说，功率谱可以看作是随机信号的频谱，参见[功率谱密度（PSD）](https://zhuanlan.zhihu.com/p/417454806)。

### 特征提取

1.  平滑功率谱的提取，包括低频带补偿和清音帧处理等过程。STRAIGHT分析阶段的一个关键步骤是进行自适应频谱分析，获取无干扰且平滑的功率谱。自适应加权谱的提取关键在于对提取出来的功率谱进行一系列的平滑和补偿。对输入信号进行：语音信号预处理-\>功率谱提取-\>低频噪声补偿-\>过平滑补偿-\>静音帧谱图的处理，最后得到自适应功率谱。

2.  非周期成分提取。

3.  通过小波时频分析的方式，提取基频轨迹。首先通过对语音信号中的基频信息进行解析，然后计算出相应的瞬时基频值，最后在频域进行谐波解析，并在频率轴进行平滑处理，获得语音信号的各个基频参数。

### 语音合成

STARIGHT采用PSOLA技术和最小相位脉冲响应相结合的方式，在合成语音时输入待合成语音的基音频率轨迹和去除了周期性的二维短时谱包络。

开源的STRAIGHT声码器大多是MATLAB实现，比如[Legacy
STRAIGHT](https://github.com/HidekiKawahara/legacy_STRAIGHT)，[StraightRepo](https://github.com/ashmanmode/StraightRepo)。在开源语音合成系统[merlin](https://github.com/CSTR-Edinburgh/merlin)中存在可用的STRAIGHT工具，参见[StraightCopySynthesis](https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/straight/copy_synthesis.sh)。