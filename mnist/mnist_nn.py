import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#MNIST数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络参数0
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNNING_RATE_BASE = 0.8
LEARNNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

#辅助函数，给定神经网络所需参数，返回输出
def inference(input_tensor, avg_class, reuse=False):
    if avg_class == None:
        with tf.variable_scope('layer1',reuse=reuse):
            weights = tf.get_variable('weights',[INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
        with tf.variable_scope('layer2',reuse=reuse):
            weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights)+biases
        return layer2
    else:
        with tf.variable_scope('layer1',reuse=reuse):
            weights = tf.get_variable('weights',[INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights))+avg_class.average(biases))
        with tf.variable_scope('layer2',reuse=reuse):
            weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, avg_class.average(weights))+avg_class.average(biases)
        return layer2

#训练模型过程
def train(mnist):
    accuracy_plot = []
    x_label = []
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None,OUTPUT_NODE), name = 'y-output')
    y = inference(x,None)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,reuse=True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope('',reuse=True):
        regularization = regularizer(tf.get_variable('layer1/weights')) + regularizer(tf.get_variable('layer2/weights'))
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNNING_RATE_BASE, global_step,mnist.train.num_examples/BATCH_SIZE,
                                               LEARNNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i%10 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s),accuracy of model is %.4f'%(i,validate_acc))
                x_label.append(i)
                accuracy_plot.append(validate_acc)
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print('After %d training step(s),test accuracy is %.4f'%(TRAINING_STEPS,test_acc))
        fig = plt.figure()
        plt.plot(x_label, accuracy_plot,'r',label='Accuracy curve')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        plt.show()
def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()