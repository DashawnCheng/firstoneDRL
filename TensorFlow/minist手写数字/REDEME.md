
使用一个神经元处理minist手写数字识别问题 
reshape将一维数组格式化 按行进行排列
为什么要采用 one hot 编码 
1.将离散特征的取值拓展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点
2.机器学习算法中，特征之间距离的计算或相似度的常用计算方法都是基于欧式空间的
3.将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理

argmax返回值是最大数的索引

#tf的正态分布的随机数
tf.random_normal()
norm =tf.random_normal([100])
with tf.Session()as sess:
	norm_data=norm.eval()
print(norm_data[:10])
#从预测问题到分类问题
#从线性回归到逻辑回归
sigmod函数
y=1/1+e^-(z)
采用梯度下降法，容易导致局部最优
逻辑回归中的损失函数一般采用对数据损失函数为二元逻辑回归
softmax示列

argmax()用法
