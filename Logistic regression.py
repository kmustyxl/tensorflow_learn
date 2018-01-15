import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

#tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])

#set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Construct model
pred = tf.nn.softmax(tf.multiply(W, x)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)), reduction_indices=1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer)
            c = sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += c/total_batch
        if (epoch+1)%display_step == 0:
            print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
    print('Optimization Finished!')
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', accuracy)

