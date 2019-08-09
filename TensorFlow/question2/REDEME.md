#多变量线性回归问题
在实际问题中需要对数据进行处理 特征数据归一化
 a/max(a)-min(b)


tensorboard 打开方式
创建方式# 为tensorborad可视化准备数据
logdir='d:/log'
#创建一个操作，用于记录损失值loss，后面在tensorborad中scalars栏可见
sum_loss_op = tf.summary.scalar("loss",loss_function)
#吧所有需要记录摘要日志文件的合并，方便一次性写入
merged = tf.summary.merge_all()
#创建摘要writer，将计算图写入摘要文件，后面在tensorborad中graphs栏可见
writer = tf.summary.FileWriter(logdir,sess.graph)

        #tensorborad可视化需要在中间加入summary_str 在数组中加入sun_loss_op
        _,summary_str,loss = sess.run([optimizer,sum_loss_op,loss_function],feed_dict={x:xs,y:ys})
        #并且写入
        writer.add_summary(summary_str,epoch)

步骤：# 读取数据文件 
#显示数据摘要描述信息
#数据准备
#把df转换为np的数组格式
#对特征数据归一化

#定义训练数据占位符
#shape 中None表示行的数量未知。
# 在实际训练师决定一次带入多少行样本，从一个样本的随机SDG到批量SDG都可以
#迭代轮次
#学习率
#定义损失函数
#创建优化器
#声明会话
#定义初始化变量的操作
# 为tensorborad可视化准备数据
#模型训练
