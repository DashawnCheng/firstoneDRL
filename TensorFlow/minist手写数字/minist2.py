import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data#读取数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import matplotlib.pyplot as plt 
import numpy as np
import time 

#模型构建
#mnist中每张图片共有28*28=784个像素点
x = tf.placeholder(tf.float32,[None,784],name="X")

#0-9一共9个数字 十个类别
y=tf.placeholder(tf.float32,[None,10],name="Y")

#隐藏层神经元数量
H1_NN = 256 

#定义变量
W1 = tf.Variable(tf.random_normal([784,H1_NN],name="W"))
b1 = tf.Variable(tf.zeros([H1_NN]),name="b")
Y1 =tf.nn.relu(tf.matmul(x,W1) + b1)
#构建输出层
W2 = tf.Variable(tf.random_normal([H1_NN,10]))
b2 = tf.Variable(tf.zeros([10]))

#定义前向传播
forward =tf.matmul(Y1,W2)+b2
#结果分类
pred =tf.nn.softmax(forward) #softmax分类

#设置训练参数
train_epochs = 10 #训练轮数
batch_size = 50#单次训练样本数
total_batch = int (mnist.train.num_examples/batch_size) #一轮训练有多少批次
display_step =1  #显示粒度
learning_rate = 0.01 #学习率
#定义损失函数 采用交叉熵损失函数
loss_function =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))

#选择优化器
optimizer =  tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
#检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况
#定义准确率
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#声明会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#模型训练
for epoch in range (train_epochs):
    for batch in range(total_batch):
        xs,ys=mnist.train.next_batch(batch_size)#读取批次数据 已经打乱了顺序
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练
    #total——batch个批次训练完成后，使用验证数据计算误差与准确率；验证集没有分批
    loss,acc=sess.run([loss_function,accuracy],
                        feed_dict={x:mnist.validation.images,
                                   y:mnist.validation.labels})
    #打印训练过程中的详细信息
    if (epoch+1) % display_step ==0:
        print("Train Epoch:",'%02d'%(epoch+1),"loss=","{:.9f}".format(loss),\
            "Accuracy=","{:.4f}".format(acc))
#显示运行总时间
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

#输出错误的预测
#compare_lists = prediction_result = np.argmax(mnist.test.labels,1)

#error_list = [i for i in range(len(compare_lists)) if comepare_lists[i]==False] 

def print_predict_errs(labels,prediction):
    count = 0
    compare_lists = (prediction == np.argmax(labels,1))
    err_lists = [i for i in range (len(compare_lists))  if compare_lists[i]== False ]
    for x in  err_lists:
        print  ("index="+str(x)+"标签值=",np.argmax(labels[x]),
                "预测值=",prediction[x])
        count = count +1
    print ("总计："+str(count))


print_predict_errs(labels=mnist.test.labels,
                    prediction = prediction_result)