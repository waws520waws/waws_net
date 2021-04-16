# 关于data_helpers.py中代码文件的解读

#### 1.导包

```python
import numpy as np
import re
import itertools
from collections import Counter
```

#### 2.数据清理

```python
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
```

#### 3.构建数据集合

* ==步骤==
  * 读取文件
  * 去除最后的空行
  * 生成句子列表
  * 将积极和消极的句子合并
  * 做数据的清洗
  * 生成标签
  * 最后返回data和lebal的集合

```python
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    
    
    positive = open(positive_data_file, "rb").read().decode('utf-8',errors="ignore")
    negative = open(negative_data_file, "rb").read().decode('utf-8',errors="ignore")
    
    # 我们从文件中可以看到最后一行是空行，所以[:-1]，去除最后一行的空行
    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]
    
    # 生成句子列表
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    
    #positive_examples = list(open(positive_data_file, "rb").read().decode('utf-8'))
    #positive_examples = [s.strip() for s in positive_examples]
    #negative_examples = list(open(negative_data_file, "rb").read().decode('utf-8'))
    #negative_examples = [s.strip() for s in negative_examples]
    # Split by words

    # 数据清理 (积极、消极数据合并)
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels

    # 生成标签
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    # 返回x和y的集合，一一对应关系
    return [x_text, y]
```

#### 4.生成batch数据，使用生成器的方式

* ==步骤==
  * 计算每个epoch中有多少个batch 
    * num_batches_per_epoch = int((len(data)-1)/batch_size) + 1  记住
  * 洗牌的方式
    * np.random.permutation的使用
  * 做两层循环
    * 针对epoch的循环
      * 每一个epoch中batch的循环
        * 使用yield的方式将每个batch的数据进行发送
          * 我们在调用的时候，生成器(懒加载)直接使用for循环的方式进行batch数据的获取

```python
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # 生成一个batch数据
    data = np.array(data)
    data_size = len(data)

    # 计算每个epoch中有多少个batches
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # 这个部分是np.random.permutation对标号进行随机
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        # 在这个部分是生成一个batches的数据
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # 以生成器的方式进行数据的生成
            yield shuffled_data[start_index:end_index]
```

