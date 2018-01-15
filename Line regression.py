import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# a = tf.constant(2)
# b = tf.constant(3)
# with tf.Session() as sess:
#     print('a = 2, b = 3')
#     print('Addition with constants: %d' %sess.run(a+b))
#     print('Multiplication with constants: %d' %sess.run(a*b))

# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)
# add = tf.add(a,b)
# mul = tf.multiply(a,b)
# with tf.Session() as sess:
#     print('Addition with variables: %i'%sess.run(add, feed_dict={a:2,b:3}))
#     print('Multiplication with variables: %i'%sess.run(mul, feed_dict={a:2, b:3}))
# matrix1 = tf.constant([[3,3]])
# matrix2 = tf.constant([[2],[2]])
# product = tf.matmul(matrix1, matrix2)
# with tf.Session() as sess:
#     print(sess.run(product))

rng = np.random

#Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

#Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

#tf Graph Input
X = tf.placeholder('float')
Y = tf.placeholder('float')

#Create Model

#Set model weights
W = tf.Variable(np.random.randn(),name='weight')
b = tf.Variable(np.random.randn(),name='bias')

#Constant a linear model
activation = tf.add(tf.multiply(W,X), b)

#Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initializer the variables
init = tf.initialize_all_variables()
step = []
costt = []
#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
        if training_epochs%display_step == 0:
            step.append(epoch)
            costt.append(sess.run(cost, feed_dict={X:train_X, Y:train_Y}))
            print('epoch: %04d'%(epoch+1),'cost=','{:.9f}'.format(sess.run(cost, feed_dict={X:train_X, Y:train_Y})), 'W=',sess.run(W),'b=', sess.run(b))
    print('Optimization Finished!')
    print('cost = ',sess.run(cost,feed_dict={X:train_X, Y:train_Y}),'W=',sess.run(W),'b=',sess.run(b))
    fig = plt.figure()
    fig1 = fig.add_subplot(121)
    fig1.plot(train_X, train_Y, 'ro', label='Original data')
    fig1.plot(train_X, sess.run(W)*train_X+sess.run(b),label='Fitted line')
    fig1.legend()
    fig2 = fig.add_subplot(122)
    fig2.plot(step,costt)
    plt.show()
