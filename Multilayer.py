from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name="Weight")  # stddev = odchylenie standardowe
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="Bias")
    return tf.Variable(initial)


def conv2d(x, W, name="conv"):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')  # zero padding to valid (nie pomylić),SAME powoduje ,że output jest rozmiaru inputu ,czyli np dla 5x5 paddingo to [1,2,2,1]
    return conv


def max_pool_2x2(x, name="max_pool"):
    tmp = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')  # max pool 2x2 ksize to rozmiar filtra =2x2 strides to przesunięcie
    return tmp


def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="X")  # placeholder na obrazy
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="LABELS")  # placeholder na labele do obrazów

    # tworzenie sieci
    # konwolucje
    ##ALBO 2X 5X5
    # with tf.name_scope("conv1"):
    #    w_conv1 = weight_variable([5, 5, 1, 32])
    #    b_conv1 = bias_variable([32])
    #    x_image = tf.reshape(x, [-1, 28, 28, 1]) #-1 (się dopasuje, nx28x28x1 bo czarno biale gdzie n = size of batch / size of 28x28x1)
    #    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) #zwykłe relu po pierwszej konwolucji

    # with tf.name_scope("max_pool_1"):
    #    h_pool1 = max_pool_2x2(h_conv1)#redukcja wymiarów, image size = 14x14

    # with tf.name_scope("conv2"):
    #    w_conv2 = weight_variable([5, 5, 32, 64])
    #    b_conv2 = bias_variable([64])
    #    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # with tf.name_scope("max_pool_2"):
    #    h_pool2 = max_pool_2x2(h_conv2)#output jest teraz 7x7x64 bo przed max poolem był 14x14x64

    # tf.summary.histogram("first 5x5 layer",w_conv1);
    # tf.summary.histogram("second 5x5 layer",w_conv2);
    ##ALBO
    ##4x 3x3
    # pierwsza zmiana z 5x5 na dwa razy 3x3

    with tf.name_scope("conv1_a"):
        W_conv1_a = weight_variable([3, 3, 1, 16])
        b_conv1_a = bias_variable([16])
        x_image = tf.reshape(x, [-1, 28, 28,
                                 1])  # -1 (się dopasuje, nx28x28x1 bo czarno biale gdzie n = size of batch / size of 28x28x1)
        h_conv1_a = tf.nn.relu(
            conv2d(x_image, W_conv1_a, name="conv1_a") + b_conv1_a)  # zwykłe relu po pierwszej konwolucji 3x3

    with tf.name_scope("conv1_b"):
        W_conv1_b = weight_variable([3, 3, 16, 32])
        b_conv1_b = bias_variable([32])
        h_conv1_b = tf.nn.relu(
            conv2d(h_conv1_a, W_conv1_b, name="conv1_b") + b_conv1_b)  # zwykłe relu po drugiej konwolucji 3x3

    with tf.name_scope("max_pool_1"):
        h_pool1 = max_pool_2x2(h_conv1_b)  # redukcja wymiarów, image size = 14x14

    # druga zmiana z 5x5 na dwa razy 3x3

    with tf.name_scope("conv2_a"):
        W_conv2_a = weight_variable([3, 3, 32, 32])
        b_conv2_a = bias_variable([32])
        h_conv2_a = tf.nn.relu(conv2d(h_pool1, W_conv2_a, name="conv2_a") + b_conv2_a)

    with tf.name_scope("conv2_b"):
        W_conv2_b = weight_variable([5, 5, 32, 64])
        b_conv2_b = bias_variable([64])
        h_conv2_b = tf.nn.relu(conv2d(h_conv2_a, W_conv2_b, name="conv2_b") + b_conv2_b)

    with tf.name_scope("max_pool_2"):
        h_pool2 = max_pool_2x2(h_conv2_b)  # output jest teraz 7x7x64 bo przed max poolem był 14x14x64

    tf.summary.histogram("First 3x3 layer", W_conv1_a);
    tf.summary.histogram("Second 3x3 layer", W_conv1_b);
    tf.summary.histogram("Third 3x3 layer", W_conv2_a);
    tf.summary.histogram("Fourth 3x3 layer", W_conv2_b);

    with tf.name_scope("fully_connected_1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1,
                                            7 * 7 * 64])  # z naszego tensora 7x7x64 tworzymy jednowymiarowy (znowy występuje -1 bo to są batche danych)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope("fully_connected_1_dropout"):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,
                                   keep_prob)  # drop out, przy trenowaniu na poziomie 50%, tensor flow sam potem normalizuje (w tym przypadku mnorzy razy 1/0.5) przy testowaniu

    with tf.name_scope("fully_connected_2"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    sess.run(tf.global_variables_initializer())

    with tf.name_scope("softmax_cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # softmax

    tf.summary.scalar("loss", cross_entropy);

    global_s = tf.Variable(0, trainable=False)

    with tf.name_scope("train"):
        starter_learning_rate = 1e-4
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_s,
                                                   100, 0.98, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_s)
        tf.summary.scalar("learning rate", learning_rate);

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy);

    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all();
    writer = tf.summary.FileWriter("/tmp/mnist/7");
    writer.add_graph(sess.graph);
    start = time.clock()
    for i in range(10000):
        batch = mnist.train.next_batch(50)  # batche po 50 obrazków(randomowo branych)
        if i % 7 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            writer.add_summary(s, i)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        if i == 2000:
            print("czas wykonania algorytmu %d", time.clock() - start)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("czas wykonania algorytmu %d", time.clock() - start)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    #   help='Directory for storing input data')
    #  FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])