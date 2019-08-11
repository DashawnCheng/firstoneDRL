import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data#读取数据
import matplotlib.pyplot as plt 
import numpy as np
from time import time 

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#模型构建
#mnist中每张图片共有28*28=784个像素点
#构建输入层
x = tf.placeholder(tf.float32,[None,784],name="X")
image_shaped_input = tf.reshape(x,[-1,28,28,1])
tf.summary.image('input',image_shaped_input,10)
#构建隐藏层
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

#tensorboard:前向输出以直方图的形式输出
tf.summary.histogram('forward',forward)

#训练模型
#0-9一共9个数字 十个类别 定义标签数据占位符
y=tf.placeholder(tf.float32,[None,10],name="Y")
#定义损失函数 采用交叉熵
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))
#tensorboard：将loss损失以标量显示
tf.summary.scalar('loss',loss_function)


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
#tensorborad讲accuracy准确率以标量显示
tf.summary.scalar('accuracy',accuracy)



startTime = time ()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#tensorborad合并所有summary
merged_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('log/',sess.graph) #创建写入符

for epoch in range (train_epochs):
    for batch in range (total_batch):
        xs,ys = mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练
        #生成summary 
        summary_str = sess.run(merged_summary_op,feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)#将summary写入文件
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
sess.close()

