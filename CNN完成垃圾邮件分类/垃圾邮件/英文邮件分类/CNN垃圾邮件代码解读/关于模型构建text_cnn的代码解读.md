# 关于模型构建text_cnn的代码解读

#### 1.导包

```python
import tensorflow as tf
import numpy as np
```

#### 2.CNN模型的构建

* 首先是初始化的过程，模型的\_\_init\_\_方法

  * 我们定义输入层，因为是静态图，先进行图构建，只有在run的时候才有数据的输入，我们使用placeholder进行占位

    * ==细节：tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 在输入的时候指定shape但是我们不知道batch的大小，所以使用None进行填充==

  * ##### Embedding layer 我们随机初始化一个shape = [vocab_size,embedding_size]的，这个部分是想从词语变成词向量

    * 构建方式中使用embedding_lookup实际上计算过程是

      * 以 [batch=4,sequence_length=5]

        * ```python
          [
              [1,2,3,4,5],
              [2,3,4,0,0],
              [1,3,4,5,0],
              [1,1,2,3,4]
          ]
          ```

      * 生成的[vocab_size = 6, embedding_size=8]

        * ```python
          [
              [0,0,0,0,0,0,0,0],
              [1,1,1,1,1,1,1,1],
              [2,2,2,2,2,2,2,2],
              [3,3,3,3,3,3,3,3],
              [4,4,4,4,4,4,4,4],
              [5,5,5,5,5,5,5,5],
          ]
          ```

      * lookup的方式是直接挑选

        * **shape [batch,sequence_length,embedding_size]**

        * ```python
          shape [batch,sequence_length,embedding_size]
          [
              [
                  [1,1,1,1,1,1,1,1],
                  [2,2,2,2,2,2,2,2],
                  [3,3,3,3,3,3,3,3],
                  [4,4,4,4,4,4,4,4],
                  [5,5,5,5,5,5,5,5]
              ],
              [
                  [2,2,2,2,2,2,2,2],
              	[3,3,3,3,3,3,3,3],
              	[4,4,4,4,4,4,4,4],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]
              ],
              [
                	[1,1,1,1,1,1,1,1],
              	[3,3,3,3,3,3,3,3],
              	[4,4,4,4,4,4,4,4],
              	[5,5,5,5,5,5,5,5],
                  [0,0,0,0,0,0,0,0]
              ],
              [
                  [1,1,1,1,1,1,1,1],
                  [1,1,1,1,1,1,1,1],
              	[2,2,2,2,2,2,2,2],
              	[3,3,3,3,3,3,3,3],
              	[4,4,4,4,4,4,4,4]
              ]
          ]
          ```

      * 因为CNN要求的输入是四维的，上面构建完embedded_chars是三维的，所以需要扩充维度

        * ```python
          # 扩充维度
          self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
          ```

  * ##### CNN的==核心部分==：卷积层的构建

    * 我们有不同的核【3，4，5】会生成不同大小的卷积后的数据

    * ```python
      # 仔细分析下面的代码，我们可以发现，我们生成的过滤器的大小（以核为3为例）[filter_size=3，embedding_size=8，num_filters=128]
      # 我们将初始化W 相当于直接生成了128个不同的3*8大小的过滤器
      # 和数据进行卷积运算，我们就可以得到128个经过一次卷积的不同特征结果
      # 最后在加上偏执b的部分即可
      filter_shape = [filter_size, embedding_size, 1, num_filters]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
      ```

    * 调用tf.nn.conv2d直接进行卷积运算

      * ```python
        # 参数：输入数据、权重参数、步长、填充策略等
        # 使用relu作为整体的卷积神经网络的激活函数
        conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        ```

    * 池化层tf.nn.max_pool的使用

      * ```python
        # 参数：
        # h:4-D `Tensor` of the format specified by `data_format`
        # ksize: A list or tuple of 4 ints. The size of the window for each dimension of the input tensor.
        # 步长
        # 填充方式
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        ```

  * 最后将池化后的结果进行收集，方便后面数据的打平

#### 3.组合所有池化的特征

* 首先先计算

```python
# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
#self.h_pool = tf.concat(pooled_outputs, 3)
self.h_pool = tf.concat(pooled_outputs,3)
self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
```

#### 4.经过Dropout层，随机的删除一些特征，防止过拟合

```python
# 第一个参数是数据，第二个参数是经过dropout后保留的数据比例dropout_keep_prob
self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
```

#### 5.最后将前面的结果输入到全连接层

* 主要需要注意下l2正则化

* 注意这样的写法
  * `self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")`

```python
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
    self.predictions = tf.argmax(self.scores, 1, name="predictions")
```

#### 6.计算损失

* 二分类问题：在这个部分我们使用的是交叉熵损失函数
* 因为我们使用了l2正则化，所以我们的总损失，就是交叉熵损失 + l2的损失的部分

```python
losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
```

#### 7.计算准确率

* 使用equal将我们的预测值和真实标签进行比较，得到准确率

```python
correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
```

## 完整代码

```python
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        # 定义模型不能直接运行，使用placeholder进行占位
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
```

## 总结：

我们的模型构建的部分都需要什么？

* 首先是模型中用到的各种超参数的准备工作
* 构建网络，不同的网络使用不同的name_space进行区分,注意各层之间的shape关系
  * input layer
  * embedding layer 
  * conv2d layer
  * max_pool
  * dropout
  * fully connected layer
* 分类问题得到预测，和真实值之间计算损失
* 分类问题得到预测，和真实值之间计算准确率

