TensorBoard 使用方式：
#tensorboard:前向输出以直方图的形式输出  定义好前向输出后可直接在后面加
tf.summary.histogram('forward',forward)
#tensorboard：将loss损失以标量显示    同样的方式直接加
tf.summary.scalar('loss',loss_function)
#tensorborad讲accuracy准确率以标量显示
tf.summary.scalar('accuracy',accuracy)
#tensorborad合并所有summary
merged_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('log/',sess.graph) #创建写入符
需要创建写入符来写入每次的变化  在会话开始之后，训练开始之前
 #生成summary  需要在训练执行批次训练之前加入这一段文字

        summary_str = sess.run(merged_summary_op,feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)#将summary写入文件
后执行即可


打开tensorborad ：
在cmd打开TensorFlow的环境 
输入tensorborad -logdir = ‘目录’ 即可

正则化 防止神经网络过拟合
    
