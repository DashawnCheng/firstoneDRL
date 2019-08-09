import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data#读取数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import matplotlib.pyplot as plt 
import numpy as np
print('训练集 train 数量：',mnist.train.num_examples,
        ',验证集 validation 数量：',mnist.validation.num_examples,
        ',测试集 test 数量',mnist.test.num_examples)

#查看train_data

print('train images shape:',mnist.train.images.shape,
        'label shape:',mnist.train.labels.shape)

len(mnist.train.images[0])
mnist.train.images[0].shape
mnist.train.images[0]
mnist.train.images[0].reshape(28,28)

#可视化image
def plot_image (image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()


plot_image(mnist.train.images[2000])

#认识标签label 独热编码 one hot 
mnist_no_one_hot = input_data.read_data_sets("MNIST_data",one_hot=False)

print (mnist_no_one_hot.train.labels[0:10])
#读取验证数据
print('vakudation images:',mnist.validation.images.shape,
       'labels:',mnist.validation.labels.shape)
#读取测试集数据
print('test images:',mnist.test.images.shape,
      'labels: ',mnist.test.labels.shape)
#批量读取数据
batch_images_xs,batch_labels_ys=\
    mnist.train.next_batch(batch_size=10)
print(batch_labels_ys)

#模型构建
#mnist中每张图片共有28*28=784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")

#0-9一共9个数字 十个类别
y=tf.placeholder(tf.float32,[None,10],name="Y")

#定义变量
W = tf.Variable(tf.random_normal([784,10],name="W"))
b = tf.Variable(tf.zeros([10]),name="b")

#定义前向传播
forward =tf.matmul(x,W)+b
#结果分类
pred =tf.nn.softmax(forward) #softmax分类

#设置训练参数
train_epochs = 50 #训练轮数
batch_size = 100#单次训练样本数
total_batch = int (mnist.train.num_examples/batch_size) #一轮训练有多少批次
display_step =1  #显示粒度
learning_rate = 0.01 #学习率
#定义损失函数 采用交叉熵损失函数
loss_function =tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

#选择优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#定义准确率
#准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#声明会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#模型训练
for epoch in range (train_epochs):
    for batch in range(total_batch):
        xs,ys=mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练
    #total——batch个批次训练完成后，使用验证数据计算误差与准确率；验证集没有分批
    loss,acc=sess.run([loss_function,accuracy],
                        feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    #打印训练过程中的详细信息
    if (epoch+1) % display_step ==0:
        print("Train Epoch:",'%02d'%(epoch+1),"loss=","{:.9f}",format(loss),\
            "Accuracy=","{:.4f}".format(acc))
print("Train Finished!")

#评估模型
accu_test =     sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy",accu_test)

#模型的应用以及可视化
prediction_result= sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})
#查看预测结果
prediction_result[0:10]
#定义可视化函数
def plot_images_labels_prediction(images,#图像列表
                                  labels,#标签列表
                                  prediction,#预测值列表
                                  index,#从第index个开始显示
                                  num=10):#缺省一次显示10幅
    fig = plt.gcf()# 获取当前图表
    fig.set_size_inches(10,12)#1英寸等于2.54cm
    if num>25:
        num=25          #最多显示25个子图
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)#获取当前要处理的子图
        ax.imshow(np.reshape(images[index],(28,28)), #显示第index个图像
                            cmap='binary')
        title = "label=" + str(np.argmax(labels[index])) #构建该图上要显示的
        if len(prediction)>0:
            title += "predict=" + str(prediction[index])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index +=1
    plt.show()

        
plot_images_labels_prediction(mnist.test.images,mnist.test.labels,prediction_result,10,10)