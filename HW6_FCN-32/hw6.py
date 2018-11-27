import tensorflow as tf
import numpy as np
import input_data

LEARNING_RATE = 0.001
MOMENTUM = 0.99
EPOCH = 30
TRAING_SIZE = 199


def FCN_model(input):
    with tf.variable_scope('layer1'):
        conv1_weights = tf.get_variable('weight', [3, 3, 3, 64],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv1_biases = tf.get_variable('bias', [64],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        conv2_weights = tf.get_variable('weight2', [3, 3, 64, 64],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv2_biases = tf.get_variable('bias2', [64],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer2'):
        pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer3'):
        conv3_weights = tf.get_variable('weight', [3, 3, 64, 128],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv3_biases = tf.get_variable('bias', [128],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        conv4_weights = tf.get_variable('weight2', [3, 3, 128, 128],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv4_biases = tf.get_variable('bias2', [128],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    with tf.name_scope('layer4'):
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer5'):
        conv5_weights = tf.get_variable('weight', [3, 3, 128, 256],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv5_biases = tf.get_variable('bias', [256],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv5 = tf.nn.conv2d(pool2, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        conv6_weights = tf.get_variable('weight2', [3, 3, 256, 256],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv6_biases = tf.get_variable('bias2', [256],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))
        conv7_weights = tf.get_variable('weight3', [3, 3, 256, 256],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv7_biases = tf.get_variable('bias3', [256],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv7 = tf.nn.conv2d(relu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))
    with tf.name_scope('layer6'):
        pool3 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer7'):
        conv8_weights = tf.get_variable('weight', [3, 3, 256, 512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv8_biases = tf.get_variable('bias', [512],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv8 = tf.nn.conv2d(pool3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases))
        conv9_weights = tf.get_variable('weight2', [3, 3, 512, 512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv9_biases = tf.get_variable('bias2', [512],
                                       initializer=tf.contrib.layers.xavier_initializer())
        conv9 = tf.nn.conv2d(relu8, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))
        conv10_weights = tf.get_variable('weight3', [3, 3, 512, 512],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv10_biases = tf.get_variable('bias3', [512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv10 = tf.nn.conv2d(relu9, conv10_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))
    with tf.name_scope('layer8'):
        pool4 = tf.nn.max_pool(relu10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer9'):
        conv11_weights = tf.get_variable('weight', [3, 3, 512, 512],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv11_biases = tf.get_variable('bias', [512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv11 = tf.nn.conv2d(pool4, conv11_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu11 = tf.nn.relu(tf.nn.bias_add(conv11, conv11_biases))
        conv12_weights = tf.get_variable('weight2', [3, 3, 512, 512],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv12_biases = tf.get_variable('bias2', [512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv12 = tf.nn.conv2d(relu11, conv12_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu12 = tf.nn.relu(tf.nn.bias_add(conv12, conv12_biases))
        conv13_weights = tf.get_variable('weight3', [3, 3, 512, 512],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv13_biases = tf.get_variable('bias3', [512],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv13 = tf.nn.conv2d(relu12, conv13_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu13 = tf.nn.relu(tf.nn.bias_add(conv13, conv13_biases))
    with tf.name_scope('layer10'):
        pool4 = tf.nn.max_pool(relu13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('layer11'):
        conv14_weights = tf.get_variable('weight', [7, 7, 512, 4096],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv14_biases = tf.get_variable('bias', [4096],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv14 = tf.nn.conv2d(pool4, conv14_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu14 = tf.nn.relu(tf.nn.bias_add(conv14, conv14_biases))

        conv15_weights = tf.get_variable('weight2', [1, 1, 4096, 4096],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv15_biases = tf.get_variable('bias2', [4096],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv15 = tf.nn.conv2d(relu14, conv15_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu15 = tf.nn.relu(tf.nn.bias_add(conv15, conv15_biases))

        conv16_weights = tf.get_variable('weight3', [1, 1, 4096, 1],
                                         initializer=tf.contrib.layers.xavier_initializer())
        conv16_biases = tf.get_variable('bias3', [1],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv16 = tf.nn.conv2d(relu15, conv16_weights, strides=[1, 1, 1, 1], padding='SAME')

        output = tf.layers.conv2d_transpose(tf.nn.bias_add(conv16, conv16_biases), 1, 64, strides=32, padding='SAME',name="final-output")
        pred_output = tf.sigmoid(output, name="final-pred")
        return pred_output, output



def IoU(pred, reality):
    TP = 0
    FP = 0
    FN = 0
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.reshape(pred.astype(int), [-1])
    reality = np.reshape(reality.astype(int), [-1])
    for i in range(len(pred)):
        if pred[i] == 1 and reality[i] == 1:
            TP += 1
        elif pred[i] != 1 and reality[i] == 1:
            FN += 1
        elif pred[i] == 1 and reality[i] != 1:
            FP += 1
    # print(str(pred[i]) + ":" + str(reality[i]))
    print(str(TP) + ":" + str(FP) + ":" + str(FN))

    return TP * 1.0 / (TP + FN + FP)


if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, 352, 1216, 3], name='x-input')

    y = tf.placeholder(tf.float32, [None, 352, 1216, 1], name='y-input')

    with tf.Session() as sess:
        dataset = input_data.Dataset()
        tf_pred, logits = FCN_model(x)
        print(tf_pred.shape)
        zeros = tf.zeros_like(y)
        mask = tf.greater_equal(y, zeros)

        after_mask_y = tf.cast(tf.boolean_mask(y, mask), tf.int32)
        after_mask_logits = tf.boolean_mask(logits, mask)
        print(after_mask_y.shape)
        print(after_mask_logits.shape)

        loss_operation = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(after_mask_y, tf.float32), logits=after_mask_logits))
        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
        grads = optimizer.compute_gradients(loss_operation)
        training_operation = optimizer.apply_gradients(grads)

        correct_prediction = tf.equal(tf.cast(y, tf.int32), tf.cast(tf_pred, tf.int32))

        train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sess.run(tf.global_variables_initializer())
        cur_epoch = 0
        loss_text = []
        valid_loss_text = []
        valid_IOU = []
        IoU_text = []
        for i in range(EPOCH * TRAING_SIZE):
            img, label = dataset.next_batch(1)
            pred, loss, trainer, acc, printone, printtwo = sess.run(
                [tf_pred, loss_operation, training_operation, train_accuracy, after_mask_y, after_mask_logits],
                feed_dict={x: img, y: label})
            loss_text.append(loss)
            IoU_text.append(IoU(pred, label))

            if (i + 1) % 99 == 0:
                loss_sum = 0.0
                iou_sum = 0.0
                valid_img = dataset.valid_image
                valid_label = dataset.valid_label
                print(valid_img.shape)
                sta = 0
                for j in range(9):
                    cur_valid_img = valid_img[5 * sta : 5 * sta + 5]
                    cur_valid_label = valid_label[5 * sta : 5 * sta + 5]
                    sta += 1
                    print(cur_valid_img.shape)
                    print(cur_valid_label.shape)
                    pred2, loss2, training_operation2 = sess.run(
                        [tf_pred, loss_operation, training_operation],
                        feed_dict={x: cur_valid_img, y: cur_valid_label})
                    loss_sum += loss2
                    iou_sum += IoU(pred2, cur_valid_label)
                    print(loss_sum)
                valid_loss_text.append(loss_sum/9.0)
                valid_IOU.append(iou_sum/9.0)
        test_image = dataset.test_image
        
        for i in range(test_image.shape[0]):
            cur_test_image = test_image[i: i+1]
            pred = sess.run(tf_pred, feed_dict = {x: cur_test_image})
            np.savetxt('./pred_label/pred_{0}.txt'.format(i), np.reshape(pred, -1), fmt='%.3f')
            np.savetxt('./pred_label/result_label_{0}.txt'.format(i), np.reshape(dataset.test_label[i : i+1], -1), fmt='%d')


        np.savetxt('./loss.txt', loss_text, fmt='%.3f')
        np.savetxt('./valid_loss.txt', valid_loss_text, fmt='%.3f')
        np.savetxt('./IOU.txt', IoU_text, fmt='%.3f')
        np.savetxt('./valid_IOU.txt', valid_IOU, fmt='%.3f')


