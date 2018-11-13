import os
import tensorflow as tf
import numpy as np
import pickle

IMAGE_SIZE = 32


class Dataset:

    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        with open(os.path.join(os.path.dirname(__file__), "cifar-100-python/train"), 'rb') as fo:
            train_dict = pickle.load(fo, encoding='bytes')
            train_d = np.asarray(train_dict[b'data']).reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE]).transpose([0, 2, 3, 1])
        with open(os.path.join(os.path.dirname(__file__), "cifar-100-python/test"), 'rb') as fo:
            test_dict = pickle.load(fo, encoding='bytes')
            test_d = np.asarray(test_dict[b'data']).reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE]).transpose([0, 2, 3, 1])

        image_decode_jpeg = tf.image.convert_image_dtype(train_d[0:40000], dtype=tf.float32)
        image_decode_test_jpeg = tf.image.convert_image_dtype(test_d, dtype=tf.float32)
        image_decode_validation_jpeg = tf.image.convert_image_dtype(train_d[40000:], dtype=tf.float32)
        sess = tf.Session()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        prev_image = sess.run(image_decode_jpeg)
        new_image = sess.run(tf.image.flip_left_right(
            tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(image_decode_jpeg, [35, 35]), 32, 32)))
        test_image = sess.run(image_decode_test_jpeg)
        validation_image = sess.run(image_decode_validation_jpeg)
        train_image = np.append(prev_image, new_image, axis=0)
        all_d = np.append(np.append(train_image, test_image, axis=0), validation_image, axis=0)
        mean = np.mean(all_d, axis=(0, 1, 2))
        train_d = np.subtract(train_image, mean)
        test_d = np.subtract(test_image, mean)
        validation_d = np.subtract(validation_image, mean)
        self.train_d = train_d
        self.train_y = np.asarray(train_dict[b'fine_labels'])[0:40000]
        self.train_y = np.append(self.train_y, self.train_y, axis=0)
        self.train_y = self.train_y.reshape([-1])
        self.train_super_y = np.asarray(train_dict[b'coarse_labels'])[0:40000]
        self.train_super_y = np.append(self.train_super_y, self.train_super_y, axis=0)
        self.test_d = test_d
        self.test_y = np.asarray(test_dict[b'fine_labels'])
        self.test_super_y = np.asarray(test_dict[b'coarse_labels'])
        self._num_examples = self.train_d.shape[0]
        self.validation_d = validation_d
        self.validation_y = np.asarray(train_dict[b'fine_labels'])[40000:]
        self.validation_super_y = np.asarray(train_dict[b'coarse_labels'])[40000:]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self.train_d = self.train_d[idx]
            self.train_y = self.train_y[idx]
            self.train_super_y = self.train_super_y[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part_x = self.train_d[start:self._num_examples]
            data_rest_part_y = self.train_y[start:self._num_examples]
            data_rest_part_super_y = self.train_super_y[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self.train_d = self.train_d[idx0]
            self.train_y = self.train_y[idx0]
            self.train_super_y = self.train_super_y[idx0]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part_x = self.train_d[start:end]
            data_new_part_y = self.train_y[start:end]
            data_new_part_super_y = self.train_super_y[start:end]
            return np.concatenate((data_rest_part_x, data_new_part_x), axis=0), np.concatenate(
                (data_rest_part_y, data_new_part_y), axis=0), np.concatenate(
                (data_rest_part_super_y, data_new_part_super_y), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.train_d[start:end], self.train_y[start:end], self.train_super_y[start:end]
