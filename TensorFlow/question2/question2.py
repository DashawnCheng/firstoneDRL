import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.utils import shuffle

# 读取数据文件 
df = pd.read_csv("data/boston.csv",header=0)
#显示数据摘要描述信息
print (df.describe())

#数据准备
df=df.values

#把df转换为np的数组格式
df =np.array(df)

#对特征数据归一化
for i in range (12):
    df[:,i]=df[:,i]/(df[:,i].max()-df[:,i].min())

#x为前十二行的所有数据
x_data = df [:,:12]
#y为最后一列标签的数据
y_data = df [:,12]

#print (y_data.shape)

#定义训练数据占位符
#shape 中None表示行的数量未知。
# 在实际训练师决定一次带入多少行样本，从一个样本的随机SDG到批量SDG都可以
x= tf.placeholder(tf.float32,[None,12],name = "X") 
y= tf.placeholder(tf.float32,[None,1],name = "Y")

#定义了一个命名空间

with tf.name_scope("Model"):
    # w 初始化值为shape=(12,1)的随机数 列向量标准差为0.01
    w=tf.Variable(tf.random_normal([12,1],stddev=0.01),name="W")

    # b 初始化为 1.0
    b= tf.Variable(1.0,name="b")

    # w和x是矩阵相乘，用matmul，不能用mitiply或者*
    def model(x,w,b):
        return tf.matmul(x,w)+b

    pred=model(x,w,b)

#迭代轮次
train_epochs = 50
#学习率
learning_rate =0.01
#定义损失函数

with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2)) #均方误差

#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

#声明会话
sess = tf.Session()

#定义初始化变量的操作
init =tf.global_variables_initializer()

# 为tensorborad可视化准备数据
logdir='d:/log'
#创建一个操作，用于记录损失值loss，后面在tensorborad中scalars栏可见
sum_loss_op = tf.summary.scalar("loss",loss_function)
#吧所有需要记录摘要日志文件的合并，方便一次性写入
merged = tf.summary.merge_all()
#创建摘要writer，将计算图写入摘要文件，后面在tensorborad中graphs栏可见
writer = tf.summary.FileWriter(logdir,sess.graph)


sess.run(init)

lost_list = [] #用于保存loss值的列表


#模型训练
for epoch in range (train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data,y_data):
        xs = xs.reshape(1,12)
        ys = ys.reshape(1,1)

        _,summary_str,loss = sess.run([optimizer,sum_loss_op,loss_function],feed_dict={x:xs,y:ys})

        writer.add_summary(summary_str,epoch)

        loss_sum = loss_sum +loss
    #打乱数据顺序
    xvalues,yvalues = shuffle(x_data,y_data)

    b0temp=b.eval(session=sess)
    w0temo=w.eval(session=sess)
    loss_average = loss_sum/len(y_data)

    lost_list.append(loss_average)

    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temo)