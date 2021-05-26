本项目是基于huggingface的 Transformers框架下进行开发的多分类任务。

本项目是通过bert、electra三个预训练模型对下游任务进行fine-tune来实现文本多分类任务。

### 维护者

- 溜溜NLP

### 数据集

- 全网新闻分类数据(SogouCA)](http://www.sogou.com/labs/resource/ca.php)


本次实验采用的sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。

下面链接帮我们整理了各种任务的数据集，有需要的可以自行下载。

- 各种任务的数据集下载：https://github.com/CLUEbenchmark/CLUEDatasetSearch


### 模型下载

本项目只使用了两种预训练模型，各位如果想使用其他模型跑效果，可自行下载。

- bert-base-chinese: https://huggingface.co/bert-base-chinese


- chinese-electra-180g-small-discriminator: https://huggingface.co/hfl/chinese-electra-180g-small-discriminator


### 模型效果

- bert-base-chinese

模型参数: batch_size = 32, maxlen = 256, epoch=30

使用bert-base-chinese预训练模型，评估结果如下:

```
	 precision	   recall         f1-score	support
健康	0.991026919 	0.996990973	0.994	        997
军事	0.996	        0.997995992	0.996996997	998
体育	0.998993964	0.996987952	0.99798995	996
教育	0.993963783	0.990972919	0.992466097	997
汽车	0.99798995	0.99498998	0.996487707	998

accuracy	0.995587645	0.995587645	0.995587645	0.995587645
macro avg	0.995594923	0.995587563	0.99558815	4986
weighted avg	0.995594803	0.995587645	0.995588132	4986

```




- chinese-electra-180g-small-discriminator

模型参数: batch_size = 32, maxlen = 256, epoch=30

使用chinese-electra-180g-small-discriminator预训练模型，评估结果如下:

```
	 precision	   recall	  f1-score	support
体育	0.995979899	0.99497992	0.995479658	996
健康	0.964671246	0.985957874	0.975198413	997
军事	0.998991935	0.992985972	0.995979899	998
教育	0.98071066	0.96890672	0.974772957	997
汽车	0.989949749	0.986973948	0.988459609	998

accuracy	0.98596069	0.98596069	0.98596069	0.98596069
macro avg	0.986060698	0.985960887	0.985978107	4986
weighted avg	0.986062082	0.98596069	0.985978705	4986

```



### 致谢

huggingface：https://github.com/huggingface/transformers
