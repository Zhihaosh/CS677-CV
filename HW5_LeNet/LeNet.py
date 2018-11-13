import tensorflow as tf
import numpy as np
import input_data

BATCH_SIZE = 64
LEARNING_RATE = 0.001
CONV_SIZE = 5
CHANNELS = 3
CONV1_DEEPTH = 6
CONV2_DEEPTH = 16
FC1_SIZE = 120
FC2_SIZE = 84
IMAGE_SIZE = 32
NUM_LABELS = 100
epoch = 80

def lenet_model(input):
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable('weight', [CONV_SIZE, CONV_SIZE, CHANNELS, CONV1_DEEPTH],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv1_biases = tf.get_variable('bias', [CONV1_DEEPTH],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable('weight', [CONV_SIZE, CONV_SIZE, CONV1_DEEPTH, CONV2_DEEPTH],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv2_biases = tf.get_variable('bias', [CONV2_DEEPTH],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2_shape = pool2.get_shape().as_list()
        nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        reshape = tf.reshape(pool2, [tf.shape(pool2)[0], nodes])
    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc1_biases = tf.get_variable('bias', [FC1_SIZE],
                                     initializer=tf.contrib.layers.xavier_initializer())
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc2_biases = tf.get_variable('bias', [FC2_SIZE],
                                     initializer=tf.contrib.layers.xavier_initializer())
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    with tf.variable_scope('fc3'):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc3_biases = tf.get_variable('bias', [NUM_LABELS],
                                     initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(fc2, fc3_weights) + fc3_biases
        return logits


def train():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE,
                                    IMAGE_SIZE,
                                    CHANNELS], name='x-input')
    y = tf.placeholder(tf.int32, [None], name='y-input')
    logits = lenet_model(x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(y, tf.cast(tf.argmax(logits, 1), tf.int32))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    file_path = "./cifar-100-python/train"
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_accuracy_matrix = tf.metrics.mean_per_class_accuracy(y, tf.cast(tf.argmax(logits, 1), tf.int32), NUM_LABELS)
    confusion_matrix = tf.confusion_matrix(y, tf.cast(tf.argmax(logits, 1), tf.int32))
    print("Loading data: " + file_path)
    dataset = input_data.Dataset()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = []
    acces = []
    valid_str = ""
    for i in range(1562 * epoch):
        xs, ys, super_ys= dataset.next_batch(batch_size=BATCH_SIZE)
        xs = np.reshape(xs, [BATCH_SIZE, IMAGE_SIZE,
                             IMAGE_SIZE,
                             CHANNELS])
        ys = np.reshape(ys, [BATCH_SIZE])

        _, loss, accuracy = sess.run([training_operation, loss_operation, train_accuracy], feed_dict={
            x: xs, y: ys
        })
        losses.append(loss)
        acces.append(accuracy)
        if (i % 781 == 0):
            print(i)
            val_x = dataset.validation_d
            val_y = dataset.validation_y
            val_super_y = dataset.validation_super_y
            val_accuracy = sess.run(
                test_accuracy, feed_dict={
                    x: val_x, y: val_y
                })
            valid_str += "training acc:" + str(accuracy) + ", "+ "validation acc:" + str(val_accuracy)
            valid_str += "\n"
    test_x, test_y, test_super_y = dataset.test_d, dataset.test_y, dataset.test_super_y
    init_l = tf.local_variables_initializer()
    sess.run(init_l)
    f = open("validation.txt", "a")
    f.write(valid_str)
    test_accuracy_matrix, test_accuracy, confusion_matrix = sess.run([test_accuracy_matrix, test_accuracy, confusion_matrix], feed_dict={
        x: test_x, y: test_y
    })
    np.savetxt('./test_y.txt', confusion_matrix.astype(int), fmt='%d')
    np.savetxt('./loss.txt',losses, fmt='%.3f')
    np.savetxt('./acces.txt',acces, fmt='%.3f')
    print("final acc: " + str(test_accuracy))
    print("final acc per class: " + str(test_accuracy_matrix[1]))

if __name__ == "__main__":
    train()
