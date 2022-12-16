## wbdc2021
-----
### 1. Requirements
For details, see requirements.txt
- python3
- datatable==0.11.1
- gensim==4.0.1
- pandas==1.2.5
- numpy==1.19.5
- deepctr-torch==0.2.6
use pytorch_py3 environment

### 2. Catalog Structure

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
### 3. Operation process
- Please enter the pytorch_py3 environment externally first: ```source activate pytorch_py3```
- Installation Environment: ```sh init.sh```
- Go to catalog： cd /home/tione/notebook/wbdc2021-semi
- Model training： ```sh train.sh``` ( skip )
- Predict and generate results files：```sh inference.sh test_path```, where test_path is the absolute path to test_b.csv
The final submission file is named. /data/submission/result.csv

### 4. Models and Features
- Model: MMOE
- parameters: 
    - batch_size: 1024
    - emded_dim: 100
    - num_epochs: 1
    - learning_rate: 0.01
- Features:
    - 6ID
    - The user emb obtained by word2vec and the feed emb provided

### 5. Algorithm Performance
- Resource allocation: 2*V 100
- Predicted elapsed time
    - Total Forecast Hours: 461s
    - Average prediction time for 2000 samples of a single target behavior: 30.9763ms

### 6. Code Description
The model prediction section code location is as follows:

| Path | Number of rows | Content |
|-----|------|------|
|src/inference.py| 105 | pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 10) |

