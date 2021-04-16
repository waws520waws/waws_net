# 关于垃圾邮件的CNN分类的代码解读

### 一、训练前的准备工作

#### 1.导包

```python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
```

#### 2.定义全局的参数

* 不同的参数做好标记
* 格式：
  * tf.flags.DEFINE_float(name,data,info)
  * 数据类型：float、string、integer、boolean等
* 使用这些参数的时候，需要对参数进行激活
  * FLAGS = tf.flags.FLAGS      
  * FLAGS.flag_values_dict()
  * 定义了上面两行之后，我们对于参数的调用可以使用"."的方式 FLAGS.positive_data_file -->得到的值是**./data/rt-polaritydata/rt-polarity.pos**

```python
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
```

#### 3.加载数据

* 这个部分实际上是加载x数据和y标签
* 构建过程：[关于data_helpers.py中代码文件的解读](./关于data_helpers.py中代码文件的解读.md)

```python
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
```

#### 4.构建字典，统一数据输入长度

* ##### 完成事件(较为重要，需要理解)

  * 首先获取所有句子中的最长的长度
  * `learn.preprocessing.VocabularyProcessor` 文本预处理
    * 第一，将所有的单词构建成一个词汇表，生成一个词汇和id的对应关系，id从一开始
    * 第二，对输入的数据进行截断和填充处理，在我们这个部分，参数是max_document_length，所以全是填充，默认是用0填充
    * 第三，将所有数据替换成id的形式
      * **[[i am good],[he am ok]] ---->[[1,2,3],[4,2,5]]**

```python
# Build vocabulary
# 最大的文本长度，因为我们输入到网络中的数据需要统一格式
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

# x 已经变成了id序号的阵列
x = np.array(list(vocab_processor.fit_transform(x_text)))
```

#### 5.随机洗牌

* 打乱数据，可以让模型更好的学习

```python
# Randomly shuffle data
np.random.seed(10)

# 对数据进行洗牌
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
```

#### 6.划分验证集和训练集

* 我们定义一个比例，取出比例位置上的数据的索引(实际上是倒序取值)，在索引前面的是训练数据，在索引后面的是验证数据

```python
# 这个部分实际上是取验证集的多少，取法比较简单，直接“从后往前”取10%，截断的思想
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
```

### 二、训练

#### 0.开启图

```python
with tf.Graph().as_default():
```

#### 1.加载我们自己的config

```python
# 载入config
session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)

# 对session应用config
sess = tf.Session(config=session_conf)
```

#### 2.开启会话

```python
with sess.as_default():
```

#### 3.模型实例化

* 模型的构建(重头戏)
  * 构建过程：[关于data_helpers.py中代码文件的解读](./关于data_helpers.py中代码文件的解读.md)

```python
# 模型实例化
cnn = TextCNN(
    # 句子的长度，我们之前定义的max_document_length
    sequence_length=x_train.shape[1],
    # 分类的长度，一共多少类别，因为我们使用的方式是[0,1],onehot的形式
    num_classes=y_train.shape[1],
    # 词汇表的长度
    vocab_size=len(vocab_processor.vocabulary_),
    # 词向量的长度
    embedding_size=FLAGS.embedding_dim,
    # 过滤器的大小[3,4,5]
    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
    # 过滤器的数量
    num_filters=FLAGS.num_filters,
    # l2正则项
    l2_reg_lambda=FLAGS.l2_reg_lambda)
```

#### 4.固定执行步骤

* ##### 定义全局步骤

  * 主要是用于当到达一定步骤后，保存模型和输出结果

* ##### 定义优化器

  * 指定优化器，和学习率

* ##### 计算梯度和值

  * 对模型产生的损失进行计算梯度

* ##### 优化器使用梯度

  * 实际上就是，对优化器中的变量执行梯度下降

```python
# Define Training procedure 定义训练程序
# 定义全局的步骤
global_step = tf.Variable(0, name="global_step", trainable=False)
# 定义优化器和学习率
optimizer = tf.train.AdamOptimizer(1e-3)
# 计算使用优化器对损失计算梯度
# 实际上生成的是两个值，一个是梯度的值，另一个是变量的值
grads_and_vars = optimizer.compute_gradients(cnn.loss)
# 优化器对计算出来的梯度进行应用
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
```

#### 5.对上面的optimizer.compute_gradients计算产生的梯度和变量的值进行保存

* 主要是用于追踪和保存中间数据

```python
grad_summaries = []
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)
```

#### 6.记录损失和准确率等数据

* 记录
  * loss
  * accuracy
  * 还有之前的梯度的summary部分一并写入硬盘文件进行保存

```python
# Summaries for loss and accuracy,记录loss和accuracy的值，方便画图
loss_summary = tf.summary.scalar("loss", cnn.loss)
acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
```

#### 7.设置检查点

* 保存模型和和参数

```python
# 检查点文件，主要是用来记录模型和模型参数的，方便断点继续训练模型
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    # checkpoint的saver对象
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
```

#### 8.将词汇写入文件，方便后面测试的时候使用

```python
# 写入词汇
vocab_processor.save(os.path.join(out_dir, "vocab"))
```

#### 9.进行全局初始化

```python
# 进行全局变量初始化
sess.run(tf.global_variables_initializer())
```

#### 10.定义训练的步骤

* 训练的步骤
  * 构建数据字典，像网络中输入的数据
  * sess.run  
    * 想要什么结果数据可以直接在第一个参数的部分以列表的形式呈现
    * feed_dict 输入的数据

```python
# 定义训练的步骤
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    # 数据字典
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    # sess.run对数据进行训练
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    # 将summary数据写入到总结中，方便tensorboard进行展示
    train_summary_writer.add_summary(summaries, step)
```

#### 11.定义验证的步骤

* 和上面训练的步骤一样

```python
def dev_step(x_batch, y_batch, writer=None):
    """
            Evaluates model on a dev set
            """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)
```

#### 12.循环数据进行训练

**重点**

* 使用data_helpers.batch_iter构建一个生成器，用于不断的产生x_batch,y_batch
* 我们通过for循环不断的从生成器中取出数据进行训练
  * 完成训练的步骤
  * 获取当前的步数
    * 当满足一定条件，保存模型
    * 当满足一定条件，进行一轮验证

```python
# Generate batches
# 生成器---构建batches
batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
# 使用for循环从生成器中不断的取出batch数据,进行训练
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    # 获取当前的训练的步数
    current_step = tf.train.global_step(sess, global_step)

    # 达到验证步骤和模型保存的步骤，实行验证和模型的保存
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, './', global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
```

