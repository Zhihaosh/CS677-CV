import os
import cv2
import numpy as np
import pickle

IMAGE_WIDTH = 352
IMAGE_HEIGHT = 1216


class Dataset:

    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self.train_image = []
        self.train_label = []
        self.test_image = []
        self.test_label = []
        self.valid_image = []
        self.valid_label = []
        l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/image/train/"))
        l.sort()
        for name in l:
            img = np.asarray(cv2.imread(os.path.join(os.path.dirname(__file__), "./data/image/train/" + name)),
                             "float32")
            self.train_image.append(img)

        l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/label/train/"))
        l.sort()
        for name in l:
            label = pickle.load(open(os.path.join(os.path.dirname(__file__), "./data/label/train/" + name), "rb"),
                                encoding='bytes')
            label = label.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
            self.train_label.append(label)

        l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/image/test/"))
        l.sort()
        for name in l:
            img = np.asarray(cv2.imread(os.path.join(os.path.dirname(__file__), "./data/image/test/" + name)),
                             "float32")
            self.test_image.append(img)

        l = os.listdir(os.path.join(os.path.dirname(__file__), "./data/label/test/"))
        l.sort()
        for name in l:
            label = pickle.load(open(os.path.join(os.path.dirname(__file__), "./data/label/test/" + name), "rb"),
                                encoding='bytes')
            label = label.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
            self.test_label.append(label)

        self.test_image = np.asarray(self.test_image[:])
        self.test_label = np.asarray(self.test_label[:])
        self.valid_image = np.asarray(self.train_image[199:])
        self.valid_label = np.asarray(self.train_label[199:])
        self.train_image = np.asarray(self.train_image[:199])
        self.train_label = np.asarray(self.train_label[:199])
        self._num_examples = self.train_image.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            self.train_image = self.train_image[idx]
            self.train_label = self.train_label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            self._index_in_epoch = 0
            start = 0
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.train_image[start:end], self.train_label[start:end]


if __name__ == '__main__':
    dataset = Dataset()
    print(dataset.valid_image.shape)
