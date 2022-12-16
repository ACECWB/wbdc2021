## wbdc2021-preliminary-48c2b28c233f4934b362696daef770e4
-----
### 1.环境依赖
详情见requirements.txt
- python3
- datatable==0.11.1
- gensim==4.0.1
- pandas==1.2.5
- numpy==1.19.5
- deepctr-torch==0.2.6
使用pytorch_py3环境

### 2.目录结构

```sh
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements and preparing train/inference data, these data will be saved in ./data/feature
├── train.sh, script for training models
├── inference.sh, script for inference
├── src
│   ├── prepare, codes for preparing train/test dataset
|   ├── train, codes for training
|   ├── inference.py, main function for inference on test dataset
|   ├── evaluation.py, main function for evaluation 
│   ├── model, codes for model architecture
│   ├── third_party, open source third-party codes
│   ├── utils, all other codes which do not belong to previous directories
├── data
│   ├── wedata, dataset of the competition
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (n_fold_lgb and difm) 
├── config, configuration files

```
### 3.运行流程
- 请先在外部进入pytorch_py3环境: ```source activate pytorch_py3```
- 安装环境: ```sh init.sh```
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 模型训练：```sh train.sh``` ( 跳过 )
- 预测并生成结果文件：```sh inference.sh test_path```, 其中test_path为test_b.csv所在绝对路径
最终提交文件命名为./data/submission/result.csv

### 4.模型及特征
- 模型: MMOE
- 参数: 
    - batch_size: 1024
    - emded_dim: 100
    - num_epochs: 1
    - learning_rate: 0.01
- 特征:
    - 6ID
    - word2vec得到的user emb和提供的feed emb

### 5.算法性能
- 资源配置: 2*V 100
- 预测耗时
    - 总预测时长: 461s
    - 单个目标行为2000条样本的平均预测时长: 30.9763ms

### 6.代码说明
模型预测部分代码位置如下:

| 路径 | 行数 | 内容 |
|-----|------|------|
|src/inference.py| 105 | pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 10) |

