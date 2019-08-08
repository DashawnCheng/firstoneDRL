import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#设置随机数种子
np.random.seed(5)
#np生成等差序列的方法，生成100个-1，1之间的点
x_data = np.linspace(-1,1,100)
# y=2x+1 + 噪声 ，噪声的维度与x_data一致
y_data =2 *x_data +1.0+np.random.randn(*x_data.shape) * 0.4

print (np.random.randn(10))
print (x_data.shape)
print (np.random.randn(100))
plt.scatter(x_data,y_data)
plt.plot(x_data,2*x_data+1.0,color='red',linewidth=3)
plt.show()
#定义训练数据的占位符，x是特征值，y是标签值

x=tf.placeholder("float",name="x")
y=tf.placeholder("float",name="y")
#定义模型函数
def model(x,w,b):
    return tf.multiply(x,w)+b 

#构建线性函数的斜率 ,变量w
w = tf.Variable(1.0,name="w0")
#构建线性函数的截距没变量b
b = tf.Variable(0.0,name="b0")
#pred是预测值，前向计算
pred =model(x,w,b)
#迭代次数
trian_epochs = 10
#学习率
learing_rate =0.05
#定义损失函数 采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))
#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss_function)

#声明会话
sess=tf.Session()
#通过下列函数可以将所有变量初始化
init = tf.global_variables_initializer()

sess.run(init)
#开始训练 采用SGD随机梯度下降优化方法 
for epoch in range(trian_epochs):
    for xs,ys in zip(x_data,y_data):
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    plt.plot(x_data,w0temp*x_data,b0temp)
plt.show()

print ("w:",sess.run(w))
print ("b:",sess.run(b))

plt.scatter(x_data,y_data,label='Original data')
plt.plot (x_data,x_data*sess.run(w)+sess.run(b),
          label='Fitted line',color='r',linewidth=3)
plt.legend(loc=2)
plt.show()

x_test = 3.21
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"% predict)

target = 2*x_test+1.0
print("实际值：%f"%target)
