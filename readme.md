## 开发环境

Python 3.7

Pytorch 1.10

Pytorch_lightning 1.15

torchcrf 0.4



## 数据集
本程序数据来源于：[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition). 

为了能够使用seqeval工具评估模型效果，将原始数据中“M-”开头的标签处理为“I-”.