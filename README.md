# firstoneDRL
DRL学习实录


#tensorflow 

% use tensorflow to print session

hello.py

import tensorflow as tf    

#建立一个常量运算，加入到默认的计算图中
hello = tf.constant("Hello,World!")

#创建一个TensorFlow的会话
sess = tf.Session()

#获取结果
print(sess.run(hello))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
