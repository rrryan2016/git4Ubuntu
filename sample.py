# Trying tensorflow 
import tensorflow as tf 

# Trial1 - a tensorflow network 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Above 2 lines are used to fix/ignore `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2`

a = tf.constant(3,name='input_a') # constant node 
b = tf.constant(5,name='input_b')
c = tf.multiply(a,b,name='mult_c') # multi output node 
d = tf.add(a,b,name='add_d') # add output node
e = tf.add(c,d,name='add_e')
sess = tf.Session() # 申明一个tensorflow Session变量，任何一个tensorflow网络都需要一个Session变量，以便后续进行运
sess.run(e) # 运行网络，输出节点e
print(sess.run(e))

#  Can u motherfxxker feel me!!??
# I can !!

# Trial2 - visualize network data flow pic
writer = tf.summary.FileWriter('./my_graph',sess.graph)    
# 将sess的数据流图和数据和概括统计量存入执行目录下的my_graph目录，输入为sess的graph属性




