
更多开源数据参见: [低调奋进-开源数据汇总](http://yqli.tech/page/data.html)

## 中文数据集
1.  [标贝中文标准女声音库csmsc](https://www.data-baker.com/open_source.html):
    中文单说话人语音合成数据集，质量高。
2.  [THCHS-30](https://www.openslr.org/18/):
    中文多说话人数据集，原为语音识别练手级别的数据集，也可用于多说话人中文语音合成。
3.  [Free ST Chinese Mandarin Corpus](https://www.openslr.org/38/):
    855个说话人，每个说话人120句话，有对应人工核对的文本，共102600句话。
4.  [zhvoice](https://github.com/KuangDD/zhvoice):
    zhvoice语料由8个开源数据集，经过降噪和去除静音处理而成，说话人约3200个，音频约900小时，文本约113万条，共有约1300万字。
5.  [滴滴800+小时DiDiSpeech语音数据集](https://arxiv.org/abs/2010.09275):
    DiDi开源数据集，800小时，48kHz，6000说话人，存在对应文本，背景噪音干净，适用于音色转换、多说话人语音合成和语音识别，参见：https://zhuanlan.zhihu.com/p/268425880。
6.  [SpiCE-Corpus](https://github.com/khiajohnson/SpiCE-Corpus):
    SpiCE是粤语和英语会话双语语料库。
7.  [HKUST](http://www.paper.edu.cn/scholar/showpdf/MUT2IN4INTD0Exwh):
    10小时，单说话人，采样率8kHz。
8.  [AISHELL-1](https://www.aishelltech.com/kysjcp):
    170小时，400个说话人，采样率16kHz。
9.  [AISHELL-2](http://www.aishelltech.com/aishell_2):
    1000小时，1991个说话人，采样率44.1kHz。希尔贝壳开源了不少中文语音数据集，AISHELL-2是最近开源的一个1000小时的语音数据库，禁止商用。官网上还有其它领域，比如用于语音识别的4个开源数据集。
10. [AISHELL-3](https://www.aishelltech.com/aishell_3):
    85小时，218个说话人，采样率44.1kHz。
## 英文数据集

1.  [LJSpeech](https://keithito.com/LJ-Speech-Dataset/):
    英文单说话人语音合成数据集，质量较高，25小时，采样率22.05kHz。
2.  [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651):
    英文多说话人语音数据集，44小时，109个说话人，每人400句话，采样率48kHz，位深16bits。
3.  [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1):
    630个说话人，8个美式英语口音，每人10句话，采样率16kHz，位深16bits。[这里是具体下载地址](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3)，下载方法：首先下载种子，然后执行：
          ctorrent *.torrent
4.  [CMU ARCTIC](http://festvox.org/cmu_arctic/packed/):
    7小时，7个说话人，采样率16kHz。语音质量较高，可以用于英文多说话人的训练。
5.  [Blizzard-2011](https://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/):
    16.6小时，单说话人，采样率16kHz。可以从[The Blizzard
    Challenge](https://www.cstr.ed.ac.uk/projects/blizzard/)查找该比赛的相关数据，从[SynSIG](https://www.synsig.org/index.php)查找该比赛的相关信息。
6.  [Blizzard-2013](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/):
    319小时，单说话人，采样率44.1kHz。
7.  [LibriSpeech](https://www.openslr.org/12):
    982小时，2484个说话人，采样率16kHz。[OpenSLR](https://www.openslr.org/resources.php)搜集了语音合成和识别常用的语料。
8.  [LibriTTS](https://www.openslr.org/60):
    586小时，2456个说话人，采样率24kHz。
9.  [VCC 2018](https://datashare.ed.ac.uk/handle/10283/3061):
    1小时，12个说话人，采样率22.05kHz。类似的，可以从[The Voice
    Conversion Challenge
    2016](https://datashare.ed.ac.uk/handle/10283/2211)获取2016年的VC数据。
10. [HiFi-TTS](http://www.openslr.org/109/):
    300小时，11个说话人，采样率44.1kHz。
11. [TED-LIUM](https://www.openslr.org/7/): 118小时，666个说话人。
12. [CALLHOME](https://catalog.ldc.upenn.edu/LDC97S42):
    60小时，120个说话人，采样率8kHz。
13. [RyanSpeech](https://github.com/roholazandie/ryan-tts):
    10小时，单说话人，采样率44.1kHz。交互式语音合成语料。
### 情感数据集
1.  [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data):
    用于语音合成和语音转换的情感数据集。
2.  [情感数据和实验总结](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts):
    实际是情感语音合成的实验总结，包含了一些情感数据集的总结。
### 其它数据集
1.  [Opencpop](https://wenet.org.cn/opencpop): 高质量歌唱合成数据集。
2.  [好未来开源数据集](https://ai.100tal.com/dataset):
    目前主要开源了3个大的语音数据集，分别是语音识别数据集，语音情感数据集和中英文混合语音数据集，都是多说话人教师授课音频。
3.  [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut):
    日语，10小时，单说话人，采样率48kHz。
4.  [KazakhTTS](https://github.com/IS2AI/Kazakh_TTS):
    哈萨克语，93小时，2个说话人，采样率44.1/48kHz。
5.  [Ruslan](https://ruslan-corpus.github.io/):
    俄语，31小时，单说话人，采样率44.1kHz。
6.  [HUI-Audio-Corpus](https://github.com/iisys-hof/HUI-Audio-Corpus-German):
    德语，326小时，122个说话人，采样率44.1kHz。
7.  [M-AILABS](https://github.com/imdatsolak/m-ailabs-dataset):
    多语种，1000小时，采样率16kHz。
8.  [India Corpus](https://data.statmt.org/pmindia/):
    多语种，39小时，253个说话人，采样率48kHz。
9.  [MLS](http://www.openslr.org/94/):
    多语种，5.1万小时，6千个说话人，采样率16kHz。
10. [CommonVoice](https://commonvoice.mozilla.org/zh-CN/datasets):
    多语种，2500小时，5万个说话人，采样率48kHz。
11. [CSS10](https://github.com/Kyubyong/css10):
    十个语种的单说话人语音数据的集合，140小时，采样率22.05kHz。
12. [OpenSLR](https://www.openslr.org/resources.php):
    OpenSLR是一个专门托管语音和语言资源的网站，例如语音识别训练语料库和与语音识别相关的软件。迄今为止，已经有100+语音相关的语料。
13. [DataShare](https://datashare.ed.ac.uk/):
    爱丁堡大学维护的数据集汇总，包含了语音、图像等多个领域的数据集和软件，语音数据集中包括了语音合成、增强、说话人识别、语音转换等方面的内容。
14. [Speech in Microsoft Research Open
    Data](https://msropendata.com/datasets?term=speech):
    微软开源数据搜索引擎中关于语音的相关数据集。
15. [voice datasets](https://github.com/jim-schwoebel/voice_datasets):
    Github上较为全面的开源语音和音乐数据集列表，包括语音合成、语音识别、情感语音数据集、语音分离、歌唱等语料，找不到语料可以到这里看看。
16. [Open Speech
    Corpora](https://github.com/JRMeyer/open-speech-corpora):
    开放式语音数据库列表，特点是包含多个语种的语料。
17. [EMIME](https://www.emime.org/participate.html):
    包含一些TTS和ASR模型，以及一个中文/英语，法语/英语，德语/英语双语数据集。
18. [Celebrity Audio
    Extraction](https://github.com/celebrity-audio-collection/videoprocess):
    中国名人数据集，包含中国名人语音和图像数据。