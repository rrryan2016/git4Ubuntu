#-*-coding:utf-8 -*-

import tensorflow as tf 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# URL: https://cloud.tencent.com/developer/article/1031251

# Part 1: graphs in TensorFlow 
graph = tf.get_default_graph()

# Get the list of all operation 
# # graph.get_operations()

# Print the name of each operation 
for op in graph.get_operations():
	print(op.name)
# It will be empty, cause no operation add in the graph


# # Part 2: Sessions in TensorFlow 
# # Create Sessions as below 
# sess = tf.Session()
# ### codes here ###
# sess.close()

# # 打开一个会话时，要记得在结尾处关闭。或者可以用python中的with语句块，如此一来，它将会自动被关闭： 

# with tf.Session() as sess:
# 	sess.run(f)

# Part 3: Tensors in TensorFlow 
# data is saved in Tensors, similar to Multidimensional Arrays in numPy

# # constant 
# a = tf.constant(1.0)
# # 不同于Python之类的其他语言，这里并不能直接打印/访问常量的值(exp. print(a))，除非在会话中运行

# with tf.Session() as sess:
# 	print(sess.run(a))

# # variables 
# b = tf.Variable(2.0,name="test_var")

# # # 在TF中，变量需要分别进行初始化，单独初始化每个变量效率很低。但TensorFlow提供了一次性初始化所有变量的机制，具体方法如下：

# # # Version 0.11 and before
# # init_op = tf.initialize_all_variables()

# # version 0.12 and later 
# init_op = tf.global_variables_initializer()
# # # will add init_op into the default graph of TF

# # then we print the variable above \
# with tf.Session() as sess:
# 	sess.run(init_op)
# 	print(sess.run(b))

# # print operations in graph 
# graph = tf.get_default_graph()
# for op in graph.get_operations():
# 	print(op.name)

# # 你可以利用TensorBoard来可视化整个网络，TensorBoard是一个用于可视化TensorFlow图和训练过程的工具。


# # Linear Regression 


# # 占位符
# # 定义两个占位符，用于随后填充训练数据
X = tf.placeholder("float")
Y = tf.placeholder("float")

import numpy as np 

trainX = np.linspace(-1,1,101)
trainY = 3 * trainX + np.random.rand(*trainX.shape) * 0.33 


# Example 
# Modeling 
# 线性回归的模型是 y_model = w * x, 我们需要计算出w的值。首先可以初始化w为0来建立一个模型, 并且定义cost函数为(Y – y_model)的平方。TensorFlow中自带了许多优化器（Optimizer），用来在每次迭代后更新梯度，从而使cost函数最小。这里我们使用GradientDescentOptimizer以0.01的学习率来训练模型, 随后会循环运行该训练操作：
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X,w)

cost = (tf.pow(Y-y_model,2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# # Training 
# # til now, we just defined the graph, but havent do any calculation 
# # TensorFlow的变量还没有被赋值。为了真正运行定义好的图，还需要创建并运行一个会话，在此之前，可以先定义初始化所有变量的操作init：

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# # 随后我们通过向feed_dict“喂”数据来运行train_op。迭代完成之后，我们打印出最终的w值，应该接近3。



with tf.Session() as sess:
	sess.run(init)
	for i in range(100):
		for (x,y) in zip(trainX,trainY):
			sess.run(train_op, feed_dict={X:x,Y:y})
	print(sess.run(w))

# # based on the practice above, newly create a session,let's see the result 
# with tf.Session() as sess:
# 	sess.run(init)
# 	print(sess.run(w))

# # the output it 0.0. 
# # 这就是符号计算(symbolic computation)的思想, 一旦脱离了之前的会话，所有的操作都不复存在。







