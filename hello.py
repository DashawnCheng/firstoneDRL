import tensorflow as tf    

hello = tf.constant("Hello,World!")

sess = tf.Session()

print(sess.run(hello))
#计算图

node1 =tf.constant(3.0,tf.float32,name="node1")
node2 =tf.constant(4.0,tf.float32,name="node2")
node3=tf.add(node1,node2)
print(node3)
sess =tf.Session()
print("运行结果：",sess.run(node3))
sess.close() #释放资源

#张量

tens1 = tf.constant([[[1,2,2],[2,2,3]],
                     [[3,5,6],[5,4,3]],
                     [[7,0,1],[9,1,9]],
                     [[11,12,7],[1,3,14]]],name="tens1")
print (tens1)

sess = tf.Session()
print(sess.run(tens1)[1,0,0])
sess.close()

#张量的类型
a = tf.constant([1,2],name="a")
b= tf.constant([2,3],name="b")
#resute= a+b
resute =tf.add(a,b)
sess = tf.Session()
print(sess.run(resute))
sess.close()