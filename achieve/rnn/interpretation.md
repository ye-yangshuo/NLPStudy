# 代码解析

```python
class TextRNN(nn.Module):
定义类TextRNN并继承nn.Module类
```

```python
	self.rnn = nn.RNN(self.input_size,self.hidden_size) 
创建类RNN的对象
输入向量首先经过RNN循环神经网络。
参数：输入向量大小，输出向量大小
input_size:输入RNN网络的特征向量大小，特征向量大小取决于输入的单个词的特征维度。如果输入是one—hot编码，则特征维度是词典中的词总数n；如果是词向量编码n*m（n为词总数，m是设置的词向量的特维度），则特征维度是m。
hidden_size：RNN网络中隐藏层的特征维度大小，可以自行设置。
最后输出的向量的特征维度为hidden_size。
```

```python
	self.out = nn.Linear(self.hidden_size,self.worddict_len,bias=False)
创建类Linear的对象
在经过了RNN神经网络后，向量将经过线性神经网络实现具体的任务
参数：输入向量大小，输出向量大小
hidden_size：输入向量维度大小
self.worddict_len：输出向量维度大小，取决于具体应用
```

```python
	outputs , hidden = self.rnn(x,hidden)
输入：
    x：输入向量的具体形状为（batch_size，step_size,feature_size）第一个是批次，表示输入向量的个数；第二个是向量中词的个数，因为rnn是顺序输入，因此词的个数就是步长；第三个是特征维度大小
    hidden：隐藏层向量具体形状为（step_size,batch_size,feature_size)第一个由该rnn是单向或双向决定，默认为1
获取输出向量以及隐藏层状态，输出向量由（前一刻的隐藏层向量与输入向量组合）乘权重矩阵得出。
输出：
	输出向量是一个列表，因为rnn顺序输出，每输入一个就输出一个，如果输入俩个词，则会获得俩个词，最后一个词是我们需要的。
```

```python
def make_batch(sentences,word_dict,worddict_len,hidden_size):
自定义的函数：
目的是为了处理数据得到可以输入模型的向量。
```

