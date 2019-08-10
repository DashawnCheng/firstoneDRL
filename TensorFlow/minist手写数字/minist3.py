import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data#读取数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import matplotlib.pyplot as plt 
import numpy as np
from time import time 


#模型构建
#mnist中每张图片共有28*28=784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")

#0-9一共9个数字 十个类别
y=tf.placeholder(tf.float32,[None,10],name="Y")

#构建模型


#隐藏层神经元数量
H1_NN = 256 
#第二个隐藏层神经元数量
H2_NN = 64

#输入层 -第一隐藏层参数和偏置项b ,初始化各项为标准差为0.1的数
W1 = tf.Variable(tf.truncated_normal([784,H1_NN],stddev=0.1))
b1 = tf.Variable(tf.zeros([H1_NN]))

#第1隐藏层 - 第二隐藏层参数和偏置项
W2 = tf.Variable(tf.truncated_normal([H1_NN,H2_NN],stddev=0.1))
b2 = tf.Variable(tf.zeros([H2_NN]))

#第2隐藏层 -输出层参数和偏置项 
W3 = tf.Variable(tf.truncated_normal([H2_NN,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

#计算第1隐藏层结果 激活函数为relu
Y1 = tf.nn.relu(tf.matmul(x,W1)+b1)
#计算第2隐藏层结果
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+b2)
#计算输出结果
forward=tf.matmul(Y2,W3)+b3

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

#训练模型
startTime = time ()
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range (train_epochs):
    for batch in range (total_batch):
        xs,ys = mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练
    
    #total_batch个批次训练完成后，使用验证数据计算误差与准确率
    loss,acc=sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,
                                                          y:mnist.validation.labels})
    if(epoch+1)%display_step == 0:
        print("train_epochs:",'%02d'%(epoch+1),
                "loss=","{:.9f}".format(loss),"accuracy=","{:.4f}".format(acc))
#显示运行总时间
duration = time()-startTime
print("Train Finished takes:","{:.2f}".format(duration))
#评估模型
accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print ("Test Accuracy",accu_test)
#进行预测
