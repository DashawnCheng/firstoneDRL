import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data#读取数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import matplotlib.pyplot as plt 
import numpy as np
from time import time 
#创建保存模型文件的目录
import os 
ckpt_dir = "./ckpt_dir/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
#还原模型
#模型构建
#mnist中每张图片共有28*28=784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")

#0-9一共9个数字 十个类别
y=tf.placeholder(tf.float32,[None,10],name="Y")

#构建模型

#定义全连接层函数
def fcn_layer(inputs, #输入数据
              input_dim, #输入神经元数量
              output_dim, #输出神经元数量
              activation=None): #激活函数
    W = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
                            #以截断正态分布的随机数初始化W
    b = tf.Variable(tf.zeros([output_dim]))
                                #以零初始化b
    XWb = tf.matmul(inputs,W)+b #建立表达式 input * W +b 
    if activation is None : #默认不使用激活函数
        outputs = XWb
    else :
        outputs = activation(XWb)
    return outputs
#隐藏层神经元数量
H1_NN = 256 
#第二个隐藏层神经元数量
H2_NN = 64
#构建隐藏层
h1 = fcn_layer(inputs = x,input_dim=784,output_dim=H1_NN,activation=tf.nn.relu)
h2 = fcn_layer(inputs = h1,input_dim=H1_NN,output_dim=H2_NN,activation=tf.nn.relu)
#构建输出层
forward = fcn_layer(inputs=h2,input_dim=H2_NN,output_dim=10,activation=None)
pred = tf.nn.softmax(forward)

#训练模型
#定义损失函数 采用交叉熵
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))
#设置训练参数
train_epochs = 10 
batch_size =50
total_batch = int(mnist.train.num_examples/batch_size)
display_step=1
learning_rate = 0.01
#选择优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
#定义准确率
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#读取模型
saver =tf.train.Saver()

sess=tf.Session()
init =tf.global_variables_initializer()
sess.run(init)
#查找是否存在存盘文件
ckpt=tf.train.get_checkpoint_state(ckpt_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)#从以保存的模型中读取参数
    print("Restore model form"+ckpt.model_checkpoint_path)

#输出模型准确率
print ("Accuracy:",accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
