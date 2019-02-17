# Law Challenge

司法人工智能挑战项目模板

为方便大家快速开发，将注意力集中在模型开发上，特提供此项目模板，现在我已经在 shallow_models 中写好了一个基于 fasttext 针对挑战中的两个多标签分类任务的例子。大家可以参考我的例子开发自己的模型，主要需要写下面四个文件：

- cfg_xxxx.py
- xxxx_data_processor.py
- xxxx_trainer.py
- xxxx_predictor.py

模板尚未完善，大家可以贡献力量并提意见，接下来我会再开发一个深度学习模型的例子。

## 模板结构

1. configs: 配置文件目录，一个模型一个配置
2. deep_models: 深度模型目录
3. shallow_models: 传统模型目录
4. example: CAIL2018给的例子
5. utils: 工具包
6. main.py 模型入口