# AI_Lab5

你好！这是《当代人工智能》课程实验五的作业仓库😊

## 代码环境

此代码使用版本为3.10.6的python语言实现，同时需要用如下命令安装相应模块：

```python
pip install -r requirements.txt
```

## 仓库文件结构
本仓库由如下所示的文件和文件夹组成：

```python
|-- lab5.py                    #code for this project
|-- 实验五数据.zip # a file folder for the data this lab needs(.zip version)
    |-- data/     # including text data and image data for this project
    |-- test_without_label.txt # the file we need to test by model 
    |-- train.txt              # the file used to train our model
|-- requirements.txt           # including all the python modules needed
|-- readme.md                  # introducing this repository
```

##代码执行流程
1. 把本仓库的所有文件下载到一个名为AI_lab5的大文件夹下，并进入：
```python
cd AI_lab5
```

2. 解压“实验五数据.zip”文件为文件夹

2. 确保自己的python版本在3.10.6左右，并下载所需模块：
```python
python --version
```
```python
pip install -r requirements.txt
```

3. 用类似如下的语句运行模型，可自行调整命令行参数（此处假设AI_lab5文件夹直接下载到C盘）：
```python
python lab5.py --epochs 10 --batch_size 16 --learning_rate 1e-5 --data_path r"C:\AI_lab5\实验五数据\data" --train_file r"C:\AI_lab5\实验五数据\train.txt" --test_file r"C:\AI_lab5\实验五数据\test_without_label.txt"
```
4. 观察实验结果即可



## 参考的库

本代码主要参考了如下两个对数据进行处理的库：

- [BERT](https://github.com/google-research/bert)

- [ResNet](https://github.com/huggingface/pytorch-image-models)

