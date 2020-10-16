#  BERT   QA匹配/文本分类

## 数据说明
- data目录下数据为2020 ccf 房产聊天问答数据。
参考链接 https://www.datafountain.cn/competitions/474/datasets



## 训练步骤
- 1.将bert的中文预训练模型放在本地，自行指定目录。下载路径为：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

- 2.train.sh脚本配置预训练模型目录，如：BERT_BASE_DIR=/home/syzong/nlp_deeplearning/chinese_L-12_H-768_A-12

- 3.执行 train.sh 开始模型训练，参数自行调整

## 模型预测
- 1.执行 predict.sh 即可，运行完会生成 test_results.tsv 文件，即为提交结果文件

## 数据构造原理如图
![image](https://github.com/syzong/images/blob/master/bert_input.png)


## requirements:
```
tensorflow >= 1.11.0   # CPU Version of TensorFlow.
tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
```

## 麻烦顺手Star 😄
