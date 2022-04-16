## 背景

前文已介绍了BiLSTM+CRF进行序列标注的理论内容，参见：[【NLP】基于BiLSTM-CRF的序列标注](https://blog.csdn.net/meiqi0538/article/details/124070334?spm=1001.2014.3001.5501)，也做了：[【NLP】基于隐马尔可夫模型（HMM）的命名实体识别（NER）实现](https://blog.csdn.net/meiqi0538/article/details/124065834?spm=1001.2014.3001.5501)。下面来看看如何使用Pytorch lightning书写BiLSTM-CRF来进行命名实体识别。本程序代码已上传github：[https://github.com/Htring/BiLSTM-CRF_NER_PL](https://github.com/Htring/BiLSTM-CRF_NER_PL)

其中主要使用pytorch_lightning来组织模型的训练，使用torchtext以及pytorch_lighting对语料处理，使用seqeval来评估序列标注的结果，使用pytorch-crf来实现CRF层。

本程序使用的Python程序包，主要如下：

- python 3.7
- pytorch 1.10,
- pytorch_lightning 1.15
- pytorch-crf 0.7.2
- torchtext 0.11.0
- seqeval 1.2.2


关于本程序的讲解可参考我的博客：[【NLP】基于Pytorch lightning与BiLSTM-CRF的NER实现](https://blog.csdn.net/meiqi0538/article/details/124209678)

## 数据集
本程序数据来源于：[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition). 

为了能够使用seqeval工具评估模型效果，将原始数据中“M-”、“E-”开头的标签处理为“I-”

## 模型效果
模型在测试集上效果如下：
```text
Testing: 100%|██████████| 4/4 [00:02<00:00,  1.15it/s]
               precision    recall  f1-score   support

        CONT       1.00      1.00      1.00        28
         EDU       0.94      0.96      0.95       112
         LOC       1.00      0.67      0.80         6
        NAME       1.00      0.96      0.98       112
         ORG       0.91      0.91      0.91       553
         PRO       0.77      0.82      0.79        33
        RACE       1.00      1.00      1.00        14
       TITLE       0.94      0.93      0.93       772

   micro avg       0.93      0.93      0.93      1630
   macro avg       0.95      0.90      0.92      1630
weighted avg       0.93      0.93      0.93      1630

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'val_f1': 0.9285714285714285}
--------------------------------------------------------------------------------
Testing: 100%|██████████| 4/4 [00:06<00:00,  1.52s/it]
```

